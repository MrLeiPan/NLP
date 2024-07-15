import torch
import torch.nn as nn
from transformers import BertTokenizer, GPT2LMHeadModel,GPT2ForSequenceClassification,GPT2Model,GPT2Tokenizer
import torch.nn.functional as F

class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'GPT2'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.safetensors'  # 模型训练结果
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 16  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 35  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.gpt2_path = 'uer/gpt2-large-chinese-cluecorpussmall'
        # tokenizer = BertTokenizer.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.tokenizer = BertTokenizer.from_pretrained(self.gpt2_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = 1280
        self.prompt_length = 16 # 软提示词长度为16，共有16种意图


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(config.gpt2_path, num_labels=config.num_classes)
        for param in self.gpt2.parameters():
            param.requires_grad = False  # 冻结权重
        # self.score = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, inputs_embeds, attention_mask):
        #context = x[0]  # 输入的句子
        #mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        return logits
