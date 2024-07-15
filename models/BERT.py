import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification, BertTokenizer, BertModel
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
'''
2024/7/8 20:27
Iter:   1300,  Train Loss:  0.48,  Train Acc: 85.94%,  Val Loss:   0.6,  Val Acc: 80.04%,  Time: 0:10:49 *
Iter:   1300,  Val P: 0.768,  Val R: 72.1372%,  Val F1: 0.7396,  Val Acc: 80.0433%,  Time: 0:10:49 *


'''


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'BERT'
        self.train_path = dataset + '/data/intention/train.txt'  # 训练集
        self.dev_path = dataset + '/data/intention/dev.txt'  # 验证集
        self.test_path = dataset + '/data/intention/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/intention/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.safetensors'  # 模型训练结果
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')  # 设备
        self.state_dict = None
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = [5e-5, 2.5e-5]  # 学习率
        self.model_path = "models/bert-base-chinese"
        # tokenizer = BertTokenizer.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = 768
        self.prompt_length = 16  # 软提示词长度为16，共有16种意图
        self.hidden_dropout_prob = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.model = BertModel.from_pretrained(config.model_path)
        self.config = config
        # self.freeze_weights()
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_state_dict()

    def init_state_dict(self):
        if self.config.state_dict is not None:
            print("Loading state_dict")
            self.load_state_dict(self.config.state_dict)
        else:
            print("no state_dict loaded")


    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结权重

    def forward(self, trains, inputs_embeds=None, attention_mask=None, freeze=False):
        # context = x[0]  # 输入的句子
        # mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # 获取输入嵌入, input_ids 类型 int64
        input_ids = trains[0]
        mask = trains[2]
        '''
        outputs为BERT的输出结果，其中最重要的两个输出就是pooler_output和last_hidden_state
        pooler_output的大小为[batch_size, hidden_size],包含了整个句子的特征信息（整个输入句子的语义表示），一般用来做分类任务
        last_hidden_state大小为[batch_size, seq_len ,hidden_size],这个张量代表了输入的句子经过多层的计算后得到特征信息，可以作为下游任务的输入（解码器）
        '''
        if freeze:
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids, attention_mask=mask)
        cts_outs = self.dropout(outputs.pooler_output)
        out = self.fc(cts_outs)
        return out
