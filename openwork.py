# coding: utf-8

import torch, torchtext
import math
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result

def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


num_steps, batch_size = 35, 2
# RNN
class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size, num_hiddens=256):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        self.dense = torch.nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        """
        inputs: (batch_size, num_steps)
        X:      (num_steps, batch_size, vocab_size)
        hiddens:
        """
        X = to_onehot(inputs, self.vocab_size)      # num_steps x (batch_size, vocab_size)
        X = torch.stack(X)
        hiddens, state = self.rnn(X, state)
        # hiddens:  (num_steps, batch_size, hidden_size)
        # state:    (        1, batch_size, hidden_size)
        hiddens = hiddens.view(-1, hiddens.shape[-1])    # (num_steps*batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state


def predict_rnn(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state[0].detach_()
                    state[1].detach_()
                else:
                    state.detach_()
            (output, state) = model(X, state)  # output.shape: (num_steps*batch_size, vocab_size)
            y = torch.flatten(Y.T)             # Y.shape:      (batch_size, num_steps)
            l = loss(output, y.long())         # y.shape:      (batch_size*num_steps, 1)

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))


if __name__=="__main__":
    # predict_rnn('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    num_hiddens = 256
    for num_hiddens in range(24, 512, 12):
        model = RNNModel(vocab_size).to(device)
        print("="*20,"num_hiddens=",num_hiddens,"="*20)
        train_and_predict_rnn(model, num_hiddens, vocab_size, device,
                              corpus_indices, idx_to_char, char_to_idx,
                              num_epochs, num_steps, lr, clipping_theta,
                              batch_size, pred_period, pred_len, prefixes)

