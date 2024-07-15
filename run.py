import time
import torch
import numpy as np
from Intention_recognition import train
from importlib import import_module
import argparse
from safetensors.torch import load_file
from pytorch_pretrained.optimization import BertAdam
from utils import build_dataset, build_iterator, get_time_dif
from torch.optim import AdamW
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, help='choose a model: GPT2, BERT, BERT_SoftPrompt, '
                                                              'BART_SoftPrompt')
parser.add_argument('--save_path', type=str, required=False, help='the save path of predictions on test set')
parser.add_argument('--lr', type=float, required=False, help='选择GPU')
parser.add_argument('--device', type=str, required=False, help='选择GPU')
parser.add_argument('--load_dict', type=str, required=False, help='加载预训练权重')
args = parser.parse_args()
print(args)

# python run.py --model BERT_SoftPrompt --device cuda:6 --lr 5e-5 --load_dict BERT_01_80
if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    model_list = ['BERT', 'BERT_SoftPrompt', 'BART_SoftPrompt','GPT2']
    model_name = args.model  # gpt2
    #model_name = model_list[0]
    print("loading ", model_name)
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    if args.load_dict is not None:
        state_dict = load_file("THUCNews/saved_dict/" + args.load_dict+".safetensors")
        config.state_dict = state_dict
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = torch.device(args.device)
    if args.save_path is not None:
        config.save_path = args.save_path
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样



    start_time = time.time()
    print("Loading data...")
    train_data, dev_data= build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    # test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config)
    # print(model)
    # 'dac_predictions.npy'
    # 加载预训练权重

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    optimizer1 = BertAdam(model.parameters(),
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    train(config, model, train_iter, dev_iter,None, args, optimizer1)