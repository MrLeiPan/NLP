from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import torch.nn as nn
from accelerate import Accelerator

class EHRHuaTuo(nn.Module):
    def __init__(self,args):
        super(EHRHuaTuo, self).__init__()
        self.huatuo = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        self.huatuo_generation_config = GenerationConfig.from_pretrained(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path,use_fast=True,trust_remote_code=True)
        # self.freeze_weights()

    def freeze_weights(self):
        for param in self.huatuo.parameters():
            param.requires_grad = False  # 冻结权重

    def forward(self, input_ids, decoder_input_ids, attention_mask, decoder_attention_mask):
        summary_ids = self.huatuo.generate(input_ids)
        # 解码并打印摘要
        response = self.config.tokenizer.decode(summary_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        return response


'''
huatuo.model.embed_tokens.weight  cuda:1
huatuo.model.layers.0 ~ huatuo.model.layers.15 cuda:2
huatuo.model.layers.16 ~ huatuo.model.layers.31 cuda:3
huatuo.model.norm.weight cuda:4
huatuo.lm_head.weight cuda:4 
'''
