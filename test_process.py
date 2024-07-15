
import json
import os
from datetime import timedelta
import time
from tqdm import tqdm
import torch
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
data_dir = 'dataset/IMCS-V2-DAC'
saved_path = 'THUCNews/data'
os.makedirs(saved_path, exist_ok=True)

tags = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
    'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
    'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
    'Inform-Symptom', 'Other'
]
tag2id = {tag: idx for idx, tag in enumerate(tags)}
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def make_data(samples, path):
    out = ''
    for pid, sample in samples.items():
        for sent in sample['dialogue']:
            x = sent['speaker'] + '：' + sent['sentence']
            out += x + '\t' + str(pid) + '\t'+ str(sent['sentence_id'])+ '\n' #对于test数据集，意图设为-1
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out


def generate_data():
    test_set = load_json(os.path.join(data_dir, 'IMCS-V2_test.json'))
    make_data(test_set, os.path.join(saved_path, 'test.txt'))


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, pid,did = lin.split('\t')
                # 将文本分割成标记，例如"hello,how are you ? -> "Tokens: ['hello', ',', 'how', 'are', 'you', '?']
                # discrete_prompt = "请预测对话意图[SEP]"
                # content = discrete_prompt + content
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token + [SEP] #['CLS','hello', ',', 'how', 'are', 'you', '?','[SEP]']
                seq_len = len(token)
                mask = []
                # 将转换为它们对应的词汇表（vocabulary）中的索引（ID）['CLS','hello', ',', 'how', 'are', 'you', '?','[SEP]'] -> [101,42,63,124,64,23,23,102]
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                #将句子序列填充到统一长度35，不足35的，由[PAD]填充
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size] #超过pad_size的长度，则截断序列
                        seq_len = pad_size
                contents.append((token_ids, int(pid),int(did), seq_len, mask))
        return contents
    #返回的格式是一个列表，每个元素格式为[(带有CLS的词向量对话句子，意图id，句子长度算上CLS，掩码序列)]
    test = load_dataset(config.test_path, config.pad_size)
    return test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches #所有数据
        # 所有数据量/给定批量大小 = 总批次数
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 将数据转换为LongTensor
        token_ids = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        pid = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        did = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        return (token_ids,seq_len, mask), (pid,did)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        # 能够把所有数据都用得到
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    # 给定数据，批量大小，在cpu还是gpu上运算
    return DatasetIterater(dataset, config.batch_size, config.device)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



