import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("HuatuoGPT2-7B-4bits", use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("HuatuoGPT2-7B-4bits", device_map="auto", torch_dtype="auto", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("HuatuoGPT2-7B-4bits")
    messages = []
    messages.append({"role": "user", "content": "肚子疼怎么办？"})
    response = model.HuatuoChat(tokenizer, messages)
    print(response)
