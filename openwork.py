# coding: utf-8

import torch, torchtext
import math
import time
import random
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
savedStdout = sys.stdout


# 建立字符索引
with open("Datasets/jaychou_lyrics.txt", encoding="utf-8") as f:
    corpus_chars = f.read()
corpus_chars = corpus_chars.replace("\n", " ").replace("\r", " ")
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)   # vocab_size = 2582
corpus_indices = [char_to_idx[char] for char in corpus_chars]


# 困惑度函数
def perplexity(l_sum, n):
    return math.exp(l_sum / n)


# utils
def one_hot(x, n_class, dtype=torch.float32):
    """
    input shape :  (n,)
    output shape:  (n, n+class)
    """
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    result.scatter_(1, x.long().view(-1, 1), 1)
    return result


def to_onehot(X, n_class):
    """
    input shape :  (batch_size, num_steps)
    output shape:  num_steps x (batch_size, n_class)
    """
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """
    X shape:   (batch_size, num_steps)
    Y shape:   (batch_size, num_steps)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size
    corpus_indices = corpus_indices[: corpus_len]
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def grad_clipping(params, theta, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


# RNN
class RNNModel(torch.nn.Module):
    """
    RNN input shape   : (num_steps, batch_size, vocab_size) if batch_first=False
    RNN output shape  : (num_steps, batch_size, vocab_size) if batch_first=False
    RNN hiddens shape : (num_steps, batch_size, hidden_size) if batch_first=False
    state shape       : (num_layers * num_directions, batch_size, vocab_size)
    dense output shape: (num_steps * batch_size, vocab_size)
    """
    def __init__(self, rnn_layer):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.vocab_size = rnn_layer.input_size
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.dense = torch.nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        """
        inputs shape: (batch_size, num_steps)
        X shape:      num_steps x (batch_size, vocab_size)
        hiddens:
        """
        X = to_onehot(inputs, self.vocab_size)
        X = torch.stack(X)
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])    # (num_steps*batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state


def predict_rnn(prefix, num_chars, model, idx_to_char, char_to_idx, device=None):
    """
    batch_size = 1
    num_steps  = 1
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1,1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])



def train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
                          num_steps, batch_size,
                          num_epochs, lr, clipping_theta,
                          pred_period, pred_len, prefixes, device=None):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    epoch_perplexity, epoch_time = [], []
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        state = None
        for X, Y in data_iter:
            X = X.to(device)
            Y = Y.to(device)
            if state is not None:
                # 相邻采样使用detach函数从计算图分离隐藏状态
                if isinstance(state, tuple):
                    state[0].detach_()
                    state[1].detach_()
                else:
                    state.detach_()
            (output, state) = model(X, state)  # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.T).to(device)             # Y.shape:      (batch_size, num_steps)
            l = loss(output, y.long())         # y.shape:      (num_steps * batch_size, 1)

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        p = math.exp(l_sum / n)
        t = time.time() - start
        epoch_perplexity.append(p)
        epoch_time.append(t)
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, p, t))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, model, idx_to_char, char_to_idx, device))
    return epoch_perplexity, epoch_time


def plot_perplexities_and_times(perplexities, trainTimes, legends, title):
    """
    superPameters, batch, metics
    """
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    for perplexity in perplexities:
        plt.plot(range(1,len(perplexity)+1), perplexity)
    plt.legend(legends)
    plt.xlabel("batch")
    plt.ylabel("perplexity")
    plt.title(title)
    plt.subplot(122)
    for trainTime in trainTimes:
        plt.plot(range(1,len(trainTime)+1), trainTime)
    plt.legend(legends)
    plt.xlabel("batch")
    plt.ylabel("trainTime")
    plt.title(title)
    plt.savefig(os.path.join("log", title+".png"))
    plt.close()

def plot_perplexities_and_times(perplexity, trainTime, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(perplexity) + 1), perplexity)
    plt.xlabel("batch")
    plt.ylabel("perplexity")
    plt.title(title)
    plt.savefig(os.path.join("log", title + "-perplexity.png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(trainTime) + 1), trainTime)
    plt.xlabel("batch")
    plt.ylabel("trainTime")
    plt.title(title)
    plt.savefig(os.path.join("log", title + "-trainTime.png"))
    plt.close()

if __name__=="__main__":
    num_steps, batch_size = 35, 32
    num_epochs, lr, clipping_theta = 250, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 100, ['分开', '不分开']

    net_set = ["RNN", "GRU", "LSTM"]
    net_dict = {"RNN":torch.nn.RNN, "GRU":torch.nn.GRU, "LSTM":torch.nn.LSTM}
    num_hiddens_set = [64, 128, 256, 512]
    num_layers_set = [1, 2, 3]
    bidrectional_set = [True, False]
    batch_size_set = [50, 100, 150, 200, 250]
    temp = pd.MultiIndex.from_product([net_set, num_hiddens_set, num_layers_set, bidrectional_set, batch_size_set])
    experimental = pd.DataFrame(data=np.zeros((len(temp),2), dtype=np.float), index=temp, columns=["perplexity", "trainTime"])
    for net in net_set:
        for num_hiddens in num_hiddens_set:
            for num_layers in num_layers_set:
                for bidrectional in bidrectional_set:
                    title = f"net={net}, num_hiddens={num_hiddens}, num_layers{num_layers}, bidrectional={bidrectional}"
                    print(title)
                    rnn_layer = net_dict[net](input_size=vocab_size, hidden_size=num_hiddens, num_layers=1, bidirectional=False)
                    model = RNNModel(rnn_layer).to(device)
                    epoch_perplexity, epoch_time = train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
                                                                         num_steps, batch_size,
                                                                         num_epochs, lr, clipping_theta,
                                                                         pred_period, pred_len, prefixes, device)
                    plot_perplexities_and_times(epoch_perplexity, epoch_time, title)
                    for b in batch_size_set:
                        experimental.loc[(net,num_hiddens,num_layers, bidrectional, b)]["perplexity"] = epoch_perplexity[b-1]
                        experimental.loc[net,num_hiddens,num_layers, bidrectional, b]["trainTime"] = epoch_time[b-1]
    experimental.to_csv("openwork_log.csv")

    # RNN - bidirectional效果不好
    # num_hiddens = 256
    # rnn_layer = torch.nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, num_layers=2, bidirectional=True)
    # model = RNNModel(rnn_layer).to(device)
    # # predict_rnn('分开', 10, model, vocab_size, idx_to_char, char_to_idx, device)
    # train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                       num_steps, batch_size,
    #                       num_epochs, lr, clipping_theta,
    #                       pred_period, pred_len, prefixes, device)


    # # RNN - num_hiddens
    # print("=" * 80)
    # print("RNN - num_hiddens")
    # print("=" * 80)
    # perplexity, trainTime = [], []
    # hidden_size_seq = [64, 128, 256, 512]
    # for num_hiddens in hidden_size_seq:
    #     print("=" * 20, "num_hiddens = ", num_hiddens, "=" * 20)
    #     rnn_layer = torch.nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, num_layers=1, bidirectional=False)
    #     model = RNNModel(rnn_layer).to(device)
    #     epoch_perplexity, epoch_time = train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                                                          num_steps, batch_size,
    #                                                          num_epochs, lr, clipping_theta,
    #                                                          pred_period, pred_len, prefixes, device)
    #     perplexity.append(epoch_perplexity)
    #     trainTime.append(epoch_time)
    # plot_perplexities_and_times(perplexity, trainTime, ["hidden_size="+str(i) for i in hidden_size_seq], "RNN - num_hiddens")


    # # RNN - num_layers
    # print("=" * 80)
    # print("RNN - num_layers")
    # print("=" * 80)
    # perplexity, trainTime = [], []
    # num_layers_seq = [1, 2, 3]
    # for num_layer in num_layers_seq:
    #     print("=" * 20, "num_layer = ", num_layer, "=" * 20)
    #     rnn_layer = torch.nn.RNN(input_size=vocab_size, hidden_size=256, num_layers=num_layer, bidirectional=False)
    #     model = RNNModel(rnn_layer).to(device)
    #     epoch_perplexity, epoch_time =  train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                                                           num_steps, batch_size,
    #                                                           num_epochs, lr, clipping_theta,
    #                                                           pred_period, pred_len, prefixes, device)
    #     perplexity.append(epoch_perplexity)
    #     trainTime.append(epoch_time)
    # plot_perplexities_and_times(perplexity, trainTime, ["num_layers=" + str(i) for i in num_layers_seq], "RNN - num_layers")


    # # GRU - num_hiddens
    # print("=" * 80)
    # print("GRU - num_hiddens")
    # print("=" * 80)
    # perplexity, trainTime = [], []
    # hidden_size_seq = [64, 128, 256, 512]
    # for num_hiddens in hidden_size_seq:
    #     print("=" * 20, "num_hiddens = ", num_hiddens, "=" * 20)
    #     rnn_layer = torch.nn.GRU(input_size=vocab_size, hidden_size=num_hiddens, num_layers=1, bidirectional=False)
    #     model = RNNModel(rnn_layer).to(device)
    #     epoch_perplexity, epoch_time = train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                                                          num_steps, batch_size,
    #                                                          num_epochs, lr, clipping_theta,
    #                                                          pred_period, pred_len, prefixes, device)
    #     perplexity.append(epoch_perplexity)
    #     trainTime.append(epoch_time)
    # plot_perplexities_and_times(perplexity, trainTime, ["hidden_size="+str(i) for i in hidden_size_seq], "GRU - num_hiddens")


    # # GRU - num_layers
    # print("=" * 80)
    # print("GRU - num_layers")
    # print("=" * 80)
    # perplexity, trainTime = [], []
    # num_layers_seq = [1, 2, 3]
    # for num_layer in num_layers_seq:
    #     print("=" * 20, "num_layer = ", num_layer, "=" * 20)
    #     rnn_layer = torch.nn.GRU(input_size=vocab_size, hidden_size=256, num_layers=num_layer,
    #                              bidirectional=False)
    #     model = RNNModel(rnn_layer).to(device)
    #     epoch_perplexity, epoch_time =  train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                                                           num_steps, batch_size,
    #                                                           num_epochs, lr, clipping_theta,
    #                                                           pred_period, pred_len, prefixes, device)
    #     perplexity.append(epoch_perplexity)
    #     trainTime.append(epoch_time)
    # plot_perplexities_and_times(perplexity, trainTime, ["num_layers=" + str(i) for i in num_layers_seq], "GRU - num_layers")


    # # LSTM - num_hiddens
    # print("=" * 80)
    # print("LSTM - num_hiddens")
    # print("=" * 80)
    # perplexity, trainTime = [], []
    # hidden_size_seq = [64, 128, 256, 512]
    # for num_hiddens in hidden_size_seq:
    #     print("=" * 20, "num_hiddens = ", num_hiddens, "=" * 20)
    #     rnn_layer = torch.nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, num_layers=1, bidirectional=False)
    #     model = RNNModel(rnn_layer).to(device)
    #     epoch_perplexity, epoch_time = train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                                                          num_steps, batch_size,
    #                                                          num_epochs, lr, clipping_theta,
    #                                                          pred_period, pred_len, prefixes, device)
    #     perplexity.append(epoch_perplexity)
    #     trainTime.append(epoch_time)
    # plot_perplexities_and_times(perplexity, trainTime, ["hidden_size=" + str(i) for i in hidden_size_seq], "LSTM - num_hiddens")
    #
    #
    # # LSTM - num_layers
    # print("=" * 80)
    # print("LSTM - num_layers")
    # print("=" * 80)
    # perplexity, trainTime = [], []
    # num_layers_seq = [1, 2, 3]
    # for num_layer in num_layers_seq:
    #     print("=" * 20, "num_layer = ", num_layer, "=" * 20)
    #     rnn_layer = torch.nn.LSTM(input_size=vocab_size, hidden_size=256, num_layers=num_layer,
    #                              bidirectional=False)
    #     model = RNNModel(rnn_layer).to(device)
    #     epoch_perplexity, epoch_time = train_and_predict_rnn(model, corpus_indices, idx_to_char, char_to_idx,
    #                                                          num_steps, batch_size,
    #                                                          num_epochs, lr, clipping_theta,
    #                                                          pred_period, pred_len, prefixes, device)
    #     perplexity.append(epoch_perplexity)
    #     trainTime.append(epoch_time)
    # plot_perplexities_and_times(perplexity, trainTime, ["num_layers=" + str(i) for i in num_layers_seq], "LSTM - num_layers")