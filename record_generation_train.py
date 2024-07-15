import time
import torch
import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, help='choose a model: GPT2, BERT, BERT_SoftPrompt,BART_SoftPrompt')
parser.add_argument('--save_path', type=str, required=False, help='the save path of predictions on test set')
parser.add_argument('--lr', type=float, required=False, help='选择GPU')
parser.add_argument('--device', type=str, required=False, help='选择GPU')
parser.add_argument('--load_dict', type=str, required=False, help='加载预训练权重')
args = parser.parse_args()
print(args)




# 门诊病历生成训练
def record_generation_train():
    pass


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

if __name__ == '__main__':

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