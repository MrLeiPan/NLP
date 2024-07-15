# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内对HuggingFace有访问限制，要用镜像下载
import logging
import torch
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from utils import get_time_dif
from logUtils import Logger
from safetensors.torch import save_file
'''
基线模型的ACC
Model	Acc	
TextCNN	0.7893	
TextRNN	0.7846	
TextRCNN	0.7949	
DPCNN	0.7791	
BERT	0.8165	
ERNIE	0.8191	

'''

# 训练
def train(config, model, train_iter, dev_iter, test_iter, args, optimizer):
    log_name = config.model_name + 'train_log_' + time.strftime("%Y_%m_%d_%H_%M", time.localtime()) + '.log'
    logger = Logger(log_file='logs/'+log_name, log_level=logging.INFO)
    logger.info("model_name:" + config.model_name + " 学习率：" + str(config.learning_rate))
    model.to(config.device)
    model.train()
    # # 微调软提示词
    # optimizer = BertAdam(model.soft_prompt.parameters(),
    #                      lr=config.learning_rate[1],
    #                      warmup=0.05,
    #                      t_total=len(train_iter) * config.num_epochs)
    #
    # optimizer1 = AdamW(model.soft_prompt.parameters(), lr=config.learning_rate[1])

    epochs = []  # 存储epoch数
    losses = []  # 存储loss值
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_acc = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    start_time = time.time()
    for epoch in range(config.num_epochs):
        logger.info("Epoch: " + str(epoch))
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        total_loss = 0
        total_train_acc = 0
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         logger.info(name + " :" + str(param.data))
        for i, (trains, labels) in enumerate(train_iter):
            # 前向传播
            outputs = model(trains)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # train_acc = metrics.accuracy_score(true,predic)
            total_loss += loss.item()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                # print("predic:", predic)
                # print("true_label:", true)
                train_acc = metrics.accuracy_score(true, predic)
                total_train_acc += train_acc
                p, r, f1, dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    save_file(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = '-'
                time_dif = get_time_dif(start_time)
                msg1 = ('Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {'
                        '4:>6.2%},  Time: {5} {6}')
                msg2 = 'Iter: {:>6},  Val P: {:>5.4},  Val R: {:>6.4%},  Val F1: {:>5.4}, Time: {} {}'
                print(msg1.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print(msg2.format(total_batch, p, r, f1, time_dif, improve))
                logger.info(
                    msg1.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve) + '\n'
                    + msg2.format(total_batch, p, r, f1, dev_acc, time_dif, improve) + '\n'
                    + "predic:" + str(predic) + '\n' + "true_label:" + str(true))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        # avg_loss = total_loss / len(train_iter)
        # # avg_train_acc = total_train_acc / len(train_iter)
        # # if avg_loss < dev_best_loss:
        # #     dev_best_loss = avg_loss
        # #     torch.save(model.state_dict(), config.save_path)  # 保存模型结果
        # #     print(f'save ---- Epoch {epoch + 1}/{config.num_epochs}')
        # print(f'Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss}')
        # epochs.append(epoch + 1)
        # losses.append(avg_loss)
        # loss_epoch_curve(losses, epochs)
        logger.info("best_dev_loss:"+str(dev_best_loss)+" best_dev_acc:"+str(dev_best_acc))
    logger.close()
    # test(config, model, test_iter, args)


def test(config, model, test_iter, args):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    p, r, f1, test_acc, test_loss, test_report, test_confusion, predict_all = evaluate(config, model, test_iter,test=True)
    # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    msg = 'Test P: {:>6.4}, Test R: {:>6.4}, Test F: {:>6.4},  Test Acc: {:>6.4%}'
    # print(msg.format(test_loss, test_acc))
    print(msg.format(p, r, f1, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    if args.save_path is not None:
        print("saving predictions to {}.".format(args.save_path))
        # np.save(args.save_path, predict_all)
        np.savez(args.save_path, test_confusion=test_confusion, test_prediction=predict_all)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    p = metrics.precision_score(labels_all, predict_all, average='macro')
    r = metrics.recall_score(labels_all, predict_all, average='macro')
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # return acc, loss_total / len(data_iter), report, confusion, predict_all
        return p, r, f1, acc, loss_total / len(data_iter), report, confusion, predict_all
    # return acc, loss_total / len(data_iter)
    return p, r, f1, acc, loss_total / len(data_iter)


def loss_epoch_curve(losses, epochs):
    plt.ion()  # 打开交互式模式，允许动态更新图形
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg_Loss')
    ax.set_title('Training Loss vs. Epoch')
    # 清除当前轴上的内容，并重新绘制曲线
    ax.clear()
    ax.plot(epochs, losses, marker='o', linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg_Loss')
    ax.set_title('Training Avg_Loss vs. Epoch')
    # 绘制图形
    plt.pause(0.1)  # 稍等一会，使得图形可以更新
