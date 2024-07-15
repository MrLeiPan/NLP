import torch
import torch.nn as nn
from transformers import BartModel,BertTokenizer, BartForConditionalGeneration
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'BART_SoftPrompt'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.safetensors'  # 模型训练结果
        self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 70  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 35  # 每句话处理成的长度(短填长切)
        self.learning_rate = [5e-5, 2.5e-5]  # 学习率
        self.model_path = "fnlp/bart-base-chinese"
        # tokenizer = BertTokenizer.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = 768
        self.prompt_length = 16 # 软提示词长度为16，共有16种意图


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.model = BartModel.from_pretrained(config.model_path).to(config.device)
        self.config = config
        self.freeze_weights()
        self.soft_prompt = nn.Embedding(config.prompt_length, config.hidden_size).to(config.device)
        self.prompt_indices = torch.arange(config.prompt_length, device=config.device).unsqueeze(0)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.init_weights()

    def init_weights(self):
        # 初始化软提示词嵌入
        nn.init.normal_(self.soft_prompt.weight, std=0.02)

    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结权重

    def forward(self, trains):
        #context = x[0]  # 输入的句子
        #mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # 获取输入嵌入, input_ids 类型 int64
        input_ids = trains[0]
        mask = trains[2]
        batch_size = input_ids.size(0)
        '''
        trains结构([batch_size,词嵌入向量],[batch_size],[batch_size,词嵌入向量])
        第一个元素是训练数据，第二个元素是对应训练数据的标签，第三个元素是对应训练数据的mask
        '''
        # 跟据输入的词嵌入的batch_size构建soft-prompt
        # prompt_embeddings维度[batch_size(32),soft-prompt_size(16),hidden_size(1280)]  类型 float32
        prompt_embeddings = self.soft_prompt(self.prompt_indices).expand(batch_size, -1, -1)
        '''
        input_ids 的形状为 (batch_size, sequence_length)，其中 sequence_length 是输入序列的长度
        config.gpt2.transformer.wte() 是 GPT-2 模型的词嵌入层，将 input_ids 转换为形状为 (batch_size, sequence_length, hidden_size) 的词嵌入。
        '''
        # [batch_size(32),sequence_length(35),hidden_size(1280)]
        input_embeddings = self.model.shared(input_ids)
        # inputs_embeds = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        # 重新调整 attention mask
        #prompt_mask = torch.ones((batch_size, self.config.prompt_length), device=self.config.device, dtype=torch.int64)
        #attention_mask = torch.cat([prompt_mask, trains[2]], dim=1)

        outputs = self.model(inputs_embeds=input_embeddings, attention_mask=mask, decoder_inputs_embeds = prompt_embeddings)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        out = self.fc(cls_hidden_state)
        return out
