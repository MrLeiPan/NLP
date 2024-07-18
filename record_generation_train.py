import time
from importlib import import_module

import torch
import argparse
import json
import os
import numpy as np
import re
import rouge
import jieba
from tqdm.auto import tqdm
import collections.abc as container_abcs
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator,dispatch_model,notebook_launcher,infer_auto_device_map, init_empty_weights

from models.HuaTuo import EHRHuaTuo
from tools import compute_model

int_classes = int
string_classes = str
from transformers import get_cosine_schedule_with_warmup,set_seed

model_weight_map = {
    'huatuo.model.embed_tokens.weight': 1,
    'huatuo.model.layers.0': 2,
    'huatuo.model.layers.1': 2,
    'huatuo.model.layers.2': 2,
    'huatuo.model.layers.3': 2,
    'huatuo.model.layers.4': 2,
    'huatuo.model.layers.5': 2,
    'huatuo.model.layers.6': 2,
    'huatuo.model.layers.7': 2,
    'huatuo.model.layers.8': 2,
    'huatuo.model.layers.9': 2,
    'huatuo.model.layers.10': 2,
    'huatuo.model.layers.11': 2,
    'huatuo.model.layers.12': 2,
    'huatuo.model.layers.13': 2,
    'huatuo.model.layers.14': 2,
    'huatuo.model.layers.15': 2,
    'huatuo.model.layers.16': 3,
    'huatuo.model.layers.17': 3,
    'huatuo.model.layers.18': 3,
    'huatuo.model.layers.19': 3,
    'huatuo.model.layers.20': 3,
    'huatuo.model.layers.21': 3,
    'huatuo.model.layers.22': 3,
    'huatuo.model.layers.23': 3,
    'huatuo.model.layers.24': 3,
    'huatuo.model.layers.25': 3,
    'huatuo.model.layers.26': 3,
    'huatuo.model.layers.27': 3,
    'huatuo.model.layers.28': 3,
    'huatuo.model.layers.29': 3,
    'huatuo.model.layers.30': 3,
    'huatuo.model.layers.31': 3,
    'huatuo.model.norm.weight': 4,
    'huatuo.lm_head.weight': 4,
}

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')
def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))
class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split('\t')
            if len(cur) == 2:
                title, content = cur[0], cur[1]  # 返回的是EHR（title）和对话（content）
                D.append((title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
    return D
def create_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag = [], True
    for title, content in data:
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first') # 对话文本数据编码为ids
        if flag and term == 'train':
            flag = False
           # print(content)
        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first') # EHR数据编码为ids
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                        }

        elif term == 'dev':
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                        }

        ret.append(features)
    return ret
def prepare_data(args, data_path, tokenizer, term='train'):
    """准备batch数据
    """
    data = load_data(data_path) # 将数据划分为EHR和对话
    data = create_data(data, tokenizer, args.max_len, term) # 对EHR 和 对话文本进行编码ids和mask
    data = KeyDataset(data) # 将data包装为torch.Dataset类，这也可以配合DataLoader使用
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate,num_workers=0)
    return data

def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}

# 门诊病历生成训练


class Load_json(object):
    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

class Preprocessor(object):

    def process(sefl, title):
        x = []
        for key, value in title.items():
            x.append(key + '：' + value)
        return ''.join(x)

    def make_data(self, samples, test_prediction ,path, mode='train'):
        lines = ''
        i = 0
        with open(path, 'w', encoding='utf-8') as f:
            for pid, sample in samples.items():
                content = []
                if mode == 'test':
                    for sent in sample['dialogue']:
                        if test_prediction[i] != 15:
                            content.append(sent['speaker'] + sent['sentence'])
                        i += 1
                else:
                    for sent in sample['dialogue']:
                        if sent['dialogue_act'] != 'Other':
                            content.append(sent['speaker'] + sent['sentence'])
                content = ''.join(content)
                title1, title2 = self.process(sample['report'][0]), self.process(sample['report'][1])
                if mode == 'train':
                    lines += title1 + '\t' + content + '\n'
                    lines += title2 + '\t' + content + '\n'
                elif mode == 'dev':
                    lines += title1 + '\t' + content + '\n'
                elif mode == 'dev_for_test':
                    lines += title1 + '\t' + content + '\n'
                else:
                    lines += content + '\n'
            f.write(lines)
        if mode == 'test':
            assert i == len(test_prediction)

def record_generation_train(model, optimizer, train_data, dev_data, tokenizer, lr_scheduler, args):
    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v for k, v in cur.items()}
            prob = model(**cur)
            # mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            # prob = prob[:, :-1]
            # prob = prob.reshape((-1, prob.size(-1)))[mask]
            # labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, cur['decoder_input_ids'])
            if i % 100 == 0:
                accelerator.print("Iter {}:  Training Loss: {}".format(i, loss.item()))
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        # 验证
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k: v for k, v in feature.items() if k != 'title'}
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module.generate(max_length=args.max_len_generate,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            # print(title)
            # print(gen)
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        accelerator.print("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(args.model_dir, 'summary_model'))
            else:
                torch.save(model, os.path.join(args.model_dir, 'summary_model'))
        # torch.save(model, os.path.join(args.model_dir, 'summary_model_epoch_{}'.format(str(epoch))))


def init_argument():
    parser = argparse.ArgumentParser(description='Huatuo')
    parser.add_argument('--model',type=str, required=False, help='choose a model')
    parser.add_argument('--model_path', default='models/HuatuoGPT2-7B-4bits', type=str, required=False, help='input model path')
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
    return args

def init_data():
    loadJson = Load_json()
    data_dir = "dataset/IMCS-V2-DAC"
    train_set = loadJson.load_json(os.path.join(data_dir, 'IMCS-V2_train.json'))
    dev_set = loadJson.load_json(os.path.join(data_dir, 'IMCS-V2_dev.json'))
    test_set = loadJson.load_json(os.path.join(data_dir, 'IMCS-V2_test.json'))
    test_prediction = np.load("THUCNews/npz/huatuo_predictions.npz")['test_prediction']

    preprocess = Preprocessor()
    preprocess.make_data(train_set, test_prediction ,'THUCNews/data/EHR/train.tsv', mode='train')
    preprocess.make_data(dev_set, test_prediction , 'THUCNews/data/EHR/dev.tsv', mode='dev')
    preprocess.make_data(dev_set,test_prediction, 'THUCNews/data/EHR/dev_predict.tsv', mode='dev_for_test')
    #preprocess.make_data(test_set,test_prediction, 'THUCNews/data/EHR/predict.tsv', mode='test')



if __name__ == '__main__':

    accelerator = Accelerator()
    # step 1. 初始化参数
    args = init_argument()
    model_name = args.model
    accelerator.print("loading ", model_name)
    ehr_model = EHRHuaTuo(args)
    accelerator.print(ehr_model)
    tokenizer = ehr_model.tokenizer
    # step 2. 准备训练数据
    train_dataloader = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

    # if args.data_parallel and torch.cuda.is_available():
    #     # device_ids = range(torch.cuda.device_count())
    #     device_ids = [1, 2, 3, 4, 5, 6, 7]
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 将huatuo模型的权重分到不同的GPU上
    model_weight_map = compute_model(args.model_path)
    accelerator.print(model_weight_map)
    huatuo = dispatch_model(ehr_model.huatuo, device_map = model_weight_map)
    ehr_model.huatuo = huatuo

    # 设置优化器
    optimizer = torch.optim.Adam(ehr_model.parameters(), lr=args.lr)

    # 学习率调度程序
    num_training_steps = int(len(train_dataloader) * (args.num_epoch + 0.35))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(args.warmup_rates*num_training_steps),num_training_steps=num_training_steps)
    ehr_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(ehr_model, optimizer, train_dataloader,lr_scheduler)

    # notebook_launcher(record_generation_train,dict(model, optimizer, train_dataloader, dev_data, tokenizer, lr_scheduler , args),num_processes=3)
    record_generation_train(ehr_model, optimizer, train_dataloader, dev_data, tokenizer, lr_scheduler , args)