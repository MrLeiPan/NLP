import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #国内对HuggingFace有访问限制，要用镜像下载
from transformers import AutoConfig, AdamW, AutoTokenizer, AutoModel
import torch
import torch.nn as nn


if __name__ == '__main__':
    n_tokens = 20
    num_class = 5
    initialize_from_vocab = True
    model_path = 'sijunhe/nezha-cn-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=False)
    config = AutoConfig.from_pretrained(model_path, num_labels= num_class, force_download=False)
    config.output_hidden_states = True  # 需要设置为true才输出
    model = AutoModel.from_pretrained(model_path, config=config, force_download=False)
 
    inputs = tokenizer("May the force be", return_tensors="pt")

    # need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
    # even though it does not matter what you pad input_ids with, it's just to make HF happy
    inputs['input_ids'] = torch.cat([torch.full((1,n_tokens), 50256), inputs['input_ids']], 1)
    inputs['attention_mask'] = torch.cat([torch.full((1,n_tokens), 1), inputs['attention_mask']], 1)
    print(inputs)
    outputs = model(**inputs)