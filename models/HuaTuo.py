from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import torch.nn as nn


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'BERT'
        self.model_path = 'HuatuoGPT2-7B-4bits'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.safetensors'  # 模型训练结果
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')  # 设备
        self.state_dict = None
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.learning_rate = [5e-5, 2.5e-5]  # 学习率
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)


class EHRHuaTuo(nn.Module):
    def __init__(self, config):
        super(EHRHuaTuo, self).__init__()
        self.huatuo = AutoModelForCausalLM.from_pretrained(config.model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        self.huatuo_generation_config = GenerationConfig.from_pretrained(config.model_path)
        self.config = config


    def forward(self, input_ids):
        pass

if __name__ == '__main__':
    pass
    # model = AutoModelForCausalLM.from_pretrained("HuatuoGPT2-7B-4bits", device_map="auto", torch_dtype="auto", trust_remote_code=True)
    # model.generation_config = GenerationConfig.from_pretrained("HuatuoGPT2-7B-4bits")
    # messages = []
    # messages.append({"role": "user", "content": "肚子疼怎么办？"})
    # response = model.HuatuoChat(tokenizer, messages)
    # print(response)
