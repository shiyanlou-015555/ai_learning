# coding=utf-8

import torch
import numpy as np
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("Datasets/jaychou_lyrics.txt", encoding="utf-8") as f:
    corpus_chars = f.read()
corpus_chars = corpus_chars.replace("\n", " ").replace("\r", " ")
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)   # vocab_size = 2582
corpus_indices = [char_to_idx[char] for char in corpus_chars]


num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
def get_sru_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _two():
        return (_one((num_inputs, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    Wx = _one((num_inputs, num_hiddens))
    W_xf, b_f = _two()  # 遗忘门参数
    W_xr, b_r = _two()  # 重置门参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return torch.nn.ParameterList([Wx, W_xf, b_f, W_xr,b_r, W_hq, b_q])


def get_gru_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return torch.nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def sru(inputs, state, params):
    Wx, W_xf, b_f, W_xr, b_r, W_hq, b_q = params
    C, = state
    outputs = []
    for X in inputs:
        X_tilda = torch.matmul(X, Wx)
        F = torch.sigmoid(torch.matmul(X, W_xf) + b_f)
        R = torch.sigmoid(torch.matmul(X, W_xr) + b_r)
        C = F * C + (1-F) * X_tilda
        H = R * torch.tanh(C) + (1-R) * X_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (C,)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
num_epochs, num_steps, batch_size, lr, clipping_theta = 400, 35, 32, 1e2, 1e-2

print("="*40)
print("using sru")
print("="*40)
utils.train_and_predict_rnn(sru, get_sru_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
print("="*40)
print("using gru")
print("="*40)
utils.train_and_predict_rnn(gru, get_gru_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)

