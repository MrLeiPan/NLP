import matplotlib.pyplot as plt
import torch
from importlib import import_module

from accelerate import init_empty_weights, infer_auto_device_map
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig


def count_sentence_lengths(file_path):
    sentence_lengths = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行尾的换行符并分割句子和意图ID
            line = line.strip()
            sentence, intent_id = line.rsplit('\t', 1)

            # 计算句子长度（忽略意图ID）
            sentence_length = len(sentence)
            sentence_lengths.append(sentence_length)

    return sentence_lengths



def saveModel(model, dict_load_path,dict_save_path):
    state_dict = load_file(dict_load_path)
    model.load_state_dict(state_dict)

    torch.save(model, dict_save_path)

    model = torch.load(dict_save_path)
    print(model)


# 计算设备图
def compute_model(model_path):
    config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
    device_map = infer_auto_device_map(model)
    return device_map


if __name__ == '__main__':
    x = import_module('models.' + "BERT")
    config = x.Config("THUCNews")
    model = x.Model(config)
    saveModel(model,"THUCNews/saved_dict/BERT_02_81.safetensors","THUCNews/saved_dict/BERT_02_81_model.pth")