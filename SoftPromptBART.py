import argparse
import os
from datasets import tqdm
# 设置 CUDA_LAUNCH_BLOCKING
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.HuaTuo import EHRHuaTuo
from record_generation_train import prepare_data
from tools import compute_model

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #国内对HuggingFace有访问限制，要用镜像下载
from transformers import AutoConfig, AdamW, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator,dispatch_model,notebook_launcher,infer_auto_device_map, init_empty_weights
from torch.utils.data import DataLoader

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(300000,256)
        self.layer0 = nn.Sequential(
            nn.Linear(256,1280),
            nn.ReLU(),
            nn.Linear(1280,2560),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Linear(2560, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self,input_ids, decoder_input_ids, attention_mask, decoder_attention_mask):
        accelerate.print(f"x0 device:{input_ids.device}")
        x=self.embedding(input_ids)
        accelerate.print(f"embed:{self.embedding.weight.device}")
        accelerate.print(f"x1 device:{x.device}")
        x=self.layer0(x)
        accelerate.print(f"layer0:{self.layer0._modules.get('0').weight.device}")
        accelerate.print(f"x2 device:{x.device}")
        x=self.layer1(x)
        accelerate.print(f"layer0:{self.layer1._modules.get('0').weight.device}")
        accelerate.print(f"x3 device:{x.device}")
        return x

if __name__ == '__main__':
    model = MyModel()
    accelerate = Accelerator()
    weight_map={
        'embedding':1,
        'layer0':2,
        'layer1':3
    }
    parser = argparse.ArgumentParser(description='Huatuo')
    parser.add_argument('--model', default='models/HuatuoGPT2-7B-4bits', type=str, required=False, help='choose a model')
    parser.add_argument('--model_path', default='models/HuatuoGPT2-7B-4bits', type=str, required=False,
                        help='input model path')
    parser.add_argument('--train_data', default='THUCNews/data/EHR/train.tsv')
    parser.add_argument('--dev_data', default='THUCNews/data/EHR/dev.tsv')
    # parser.add_argument('--pretrain_model', default='THUCNews/t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='THUCNews/EHR/saved_model')
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--num_epoch', default=3, help='number of epoch')
    parser.add_argument('--batch_size', default=2, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=True)
    parser.add_argument('--max_len', default=1024, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=256, help='max length of outputs')
    args = parser.parse_args()
    model = dispatch_model(model,device_map=weight_map)
    data = torch.randint(0,10000,(50, 1024)).to("cuda:0")
    # data_loader = DataLoader(data , batch_size=2)
    # step 1. 初始化参数
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,use_fast=True,trust_remote_code=True)
    # step 2. 准备训练数据
    train_dataloader = prepare_data(args, args.train_data, tokenizer, term='train')
    model, data_loader = accelerate.prepare(model, train_dataloader)
    for epoch in range(3):
        for i,cur in enumerate(tqdm(data_loader, desc='Epoch {}:'.format(epoch))):
            cur = {k: v for k, v in cur.items()}
            # y = model(cur,"","","")
            y = model(**cur)
        print(y)


'''
    dispatch_model()：可以自己定义让模型各层的权重放置在哪一个gpu上
    accelerate.prepare():在开始训练前需要将模型、tokenizer、数据加载器等包装，这样训练的时候才能让acclerator进行管理
    
    结合上述两个方式，可以实现模型分层、训练数据在隔层流动

'''