# -*- coding:utf-8 -*-

import collections
import os
import random
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import jieba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取数据
def read_comments(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            label, comment = line.split()
            data.append([comment, int(label)])
    random.shuffle(data)
    return data
all_data = read_comments("Datasets/comments/train_shuffle.txt")
# 提交的时候用全部数据训练，调参的时候要切分训练集和预测集
train_data = all_data[:16000]
valid_data = all_data[14000:]

max_l = 19
# 预处理数据
def get_tokenized_comments(data):
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return list(text)
        # return jieba.lcut(text, cut_all=True)
    return [tokenizer(comment) for comment, _ in data]



def get_vocab_comments(data):
    tokenized_data = get_tokenized_comments(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=2)


def preprocess_comments(data, vocab):
    """因为每条评论长度不一致所以不能直接组合成小批量，我们定义preprocess_imdb函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补0来将每条评论长度固定成15。"""
      # 将每条评论通过截断或者补0，使得长度变成500
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    tokenized_data = get_tokenized_comments(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


# 创建数据集
vocab = get_vocab_comments(train_data)
print(len(vocab))
batch_size = 32
train_set = Data.TensorDataset(*preprocess_comments(train_data, vocab))
valid_set = Data.TensorDataset(*preprocess_comments(valid_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
valid_iter = Data.DataLoader(valid_set, batch_size)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)


# RNN
class BiRNN(torch.nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.dropout = nn.Dropout(p=0.1)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        dropout = self.dropout(encoding)
        outs = self.decoder(dropout)
        return outs

# 字符向量
embed_size, num_hiddens, num_layers = 30, 44, 2
# 结巴分词
# embed_size, num_hiddens, num_layers = 50, 50, 3
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)


# 准确度评价指标
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def evaluate_auc(data_iter, net, device=None):
    if device is None:
        device = list(net.parameters())[0].device
    y_true, y_hat = np.zeros(0), np.zeros(0)
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            y_hat = np.concatenate([y_hat, softmax(net(X.to(device)).detach().cpu())[:,1].numpy()])
            y_true = np.concatenate([y_true, y.cpu().numpy()])
            net.train() # 改回训练模式
    return roc_auc_score(y_true, y_hat), y_hat


# 训练函数
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_auc, _ = evaluate_auc(train_iter, net)
        test_acc = evaluate_accuracy(test_iter, net)
        test_auc, _ = evaluate_auc(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, train auc %.3f, test auc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, train_auc, test_auc, time.time() - start))


lr, num_epochs = 0.01, 3
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, valid_iter, net, loss, optimizer, device, num_epochs)


# 预测函数
def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return label.item()

predict_sentiment(net, vocab, list('鱼烧的恰到好处')) #1
predict_sentiment(net, vocab, list('团购很合适'))     #0


# 输出结果
lines =  open("Datasets/comments/test_handout.txt", encoding="utf-8").readlines()
def preprocess_comments2(data, vocab):
    """因为每条评论长度不一致所以不能直接组合成小批量，我们定义preprocess_imdb函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补0来将每条评论长度固定成10。"""
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    tokenized_data = [list(line.strip()) for line in data]
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    return features
test_X = preprocess_comments2(lines, vocab)
# test_y = torch.nn.Sigmoid()(net(test_X.to(device))[:, 1]).detach().cpu().numpy()
test_y = softmax(net(test_X.to(device))).detach().cpu()[:,1].numpy()
pd.DataFrame({"ID":range(0,len(test_y)),"Prediction":test_y}).to_csv("log/submission_random.csv", index=False)





