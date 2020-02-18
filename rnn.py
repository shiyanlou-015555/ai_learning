import torch
import torchvision
import torch.nn
import torch.nn.init
import torch.utils.data as data
# print(torch.__version__)
# print(torchvision.__version__)
# print(torch.cuda.is_available())
import numpy as np
import random
import time
import math
import nlp

import basic
import data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def one_hot(x, n_class, dtype=torch.float32):
    """
    对一维特征进行独热编码。

    :param x:类别标签向量（batch_size,1）或者（batch_size,）
    :param n_class:类别数量
    :param dtype:
    :return:独热编码矩阵（batch_size, n_class)
    """
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    """
    对时间序列数据进行独热编码，可以作为RNN模型的输入。

    :param X: 采样后的时序数据，(batch, seq_len)。
    :param n_class: 类别数量。
    :return: seq_len elements of (batch, n_class)
    """
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def get_params(num_inputs, num_hiddens, num_outputs):
    """
    初始化深RNN的模型参数。

    :param num_inputs:输入特征数量。
    :param num_hiddens: 隐藏状态特征数量。
    :param num_outputs: 输出特征数量
    :return:
    """
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return torch.nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device),


def grad_clipping(params, theta, device):
    """
    梯度裁剪，对所有的模型参数一起裁剪。

    :param params: 模型参数。
    :param theta: 阈值。
    :param device: CPU/GPU。
    :return: 裁剪后的模型参数。
    """
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def rnn(inputs, state, params):
    """
    RNN网络前向步进运算。

    :param inputs: num_steps个形状为(batch_size, vocab_size)。
    :param state: 模型隐藏状态的初始值。
    :param params: RNN网络参数，W_xh, W_hh, b_h, W_hq, b_q。
    :return: num_steps个形状为(batch_size, vocab_size)
    """
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    """
    基于RNN模型继续补全句子，一次输入一个字符，预测一个字符，因此batch_size=1，step_num=1。

    :param prefix: 起始输入序列，未预处理。
    :param num_chars: 预测长度。
    :param rnn: RNN模型。
    :param params: RNN模型参数。
    :param init_rnn_state: 模型隐藏层初始状态。
    :param num_hiddens: 隐藏单元数量。
    :param vocab_size: 语料库大小。
    :param device: GPU/CPU。
    :param idx_to_char: list。
    :param char_to_idx: dict。
    :return: 输出序列。
    """
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # X是step_nums个(batch, n_class)组成的列表。
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    """
    RNN模型的训练和预测，RNN模型训练和应用时的batch_size，step_nums不一样。
    应用时batch_size=1，step_nums=1。

    :param rnn: RNN模型。
    :param get_params: 模型参数。
    :param init_rnn_state: 隐藏层初始状态。
    :param num_hiddens: 隐藏状态特征数量。
    :param vocab_size: 语料库大小。
    :param device: CPU/GPU。
    :param corpus_indices: 时序数据列表。
    :param idx_to_char: list。
    :param char_to_idx: dict。
    :param is_random_iter: 随机采样/相邻采样。
    :param num_epochs:
    :param num_steps: 时间步长=1。
    :param lr: 学习率。
    :param clipping_theta: 梯度裁剪阈值。
    :param batch_size: 批大小。
    :param pred_period: 预测的epochs间隔。
    :param pred_len: 预测长度。
    :param prefixes: 提词。
    :return: None
    """
    if is_random_iter:
        data_iter_fn = nlp.data_iter_random
    else:
        data_iter_fn = nlp.data_iter_consecutive
    params = get_params(num_inputs, num_hiddens, num_outputs)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
            # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了使模型参数
            # 的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # num_steps*batch 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差

            l = loss(outputs, y.long())
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            basic.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


if __name__=="__main__":
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = data.load_data_jay_lyrics()
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
