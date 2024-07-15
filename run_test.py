import json
import time
import torch
import numpy as np
import itertools
from importlib import import_module
from safetensors.torch import load_file
import argparse
from safetensors.torch import load_file
from test_process import build_dataset, build_iterator, get_time_dif
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, help='choose a model: GPT2, BERT, BERT_SoftPrompt, '
                                                              'BART_SoftPrompt')
parser.add_argument('--save_path', type=str, required=False, help='the save path of predictions on test set')
parser.add_argument('--lr', type=float, required=False, help='选择GPU')
parser.add_argument('--device', type=str, required=False, help='选择GPU')
parser.add_argument('--load_dict', type=str, required=False, help='加载预训练权重')
args = parser.parse_args()
print(args)

tags = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
    'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
    'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
    'Inform-Symptom', 'Other'
]

def save_predict_to_json(predictions, save_path):
    data = {}
    # 使用 itertools.chain 将嵌套列表展平
    p_flat = list(itertools.chain(*predictions))
    # 遍历 predictions 列表
    for dialog_id, sentence_id, intent_id in p_flat:
        if dialog_id not in data:
            data[dialog_id] = {}
        data[dialog_id][sentence_id] = tags[intent_id]

    # 将字典保存为 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("数据已成功保存到 ",save_path)
def predict(model, test_iter,config,save_name):
    print(test_iter)
    model.to(config.device)
    model.eval()
    with torch.no_grad():
        predictions = []
        print("正在预测...")
        for texts, info in test_iter:
            outputs = model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu()
            p_c = torch.stack((info[0].cpu(), info[1].cpu(),predict), dim=1)
            predictions.append(p_c.tolist())

        save_predict_to_json(predictions,"THUCNews/test/"+save_name)


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
    test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config)
    # print(model)
    # state_dict = load_file("THUCNews/saved_dict/BERT_02_81_model.pth")
    # model.load_state_dict(torch.load("THUCNews/saved_dict/BERT_02_81_model.pth"))
    # print(model)
    # 'dac_predictions.npy'
    # 加载预训练权重
    predict(model, test_iter,config,model_name+"_predict.json")