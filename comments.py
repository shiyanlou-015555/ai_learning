# -*- coding:utf-8

import torch
import numpy as np
import collections
import random
import time
from sklearn.metrics import roc_auc_score

train = np.loadtxt("Datasets/comments/train_shuffle.txt", dtype=np.str, encoding="utf-8")
Y = train[:,0].astype(np.long)
X = train[:,1]
train_X, train_Y = X[:14000], Y[:14000]
test_X, test_Y = X[14000:], Y[14000:]


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(0,num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i:min(i+batch_size,num_examples)]
        yield features[j], labels[j]


class Vocab:
    def __init__(self):
        self.word2idx = {"unk":0}
        self.idx2word = ["unk"]
        self.length = 0

    def add_word(self, word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.length+1
            self.length += 1
        return self.word2idx[word]

    def __len__(self):
        return self.length+1

    def onehot_word(self, word):
        vec = torch.zeros(self.length+1)
        if word in self.idx2word:
            vec[self.word2idx[word]]=1
        else:
            vec[0]=1
        return vec

    def onehot_sentence(self, sentence):
        vecs = [self.onehot_word(w) for w in sentence]
        return torch.stack(vecs)

    def onehot_sentences(self, sentences, max_len=19):
        vecs = [self.onehot_sentence(sentence + "熵" * max(max_len - len(sentence), 0)) for sentence in sentences]
        return torch.stack(vecs)


def build_vocab(X):
    vocab = Vocab()
    for s in X:
        for w in s:
            vocab.add_word(w)
    return vocab


class RNN(torch.nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = torch.nn.Linear(64, 2)


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        out = torch.nn.Sigmoid()(out)
        return out


def train_classify(net, vocab, train_iter, test_iter, batch_size, LR=0.01, num_epochs=250):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # optimize all cnn parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, n, start = 0.0, 1, time.time()

        for step, (X, Y) in enumerate(train_iter):  # gives batch data
            print("\b"*4,step,end="")
            X = vocab.onehot_sentences(X)
            Y = torch.from_numpy(Y).long()
            output = rnn(X)  # rnn output
            _, Y_hat = torch.max(output, dim=1)
            loss = loss_func(output, Y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            train_l_sum += loss.cpu().item()
            n += Y.shape[0]

        if (epoch+1)%10 == 0:
            print('epoch %d, losses %.4f, auc %f, time %.2f sec' % (
                epoch+1, train_l_sum/n, roc_auc_score(Y.numpy(), Y_hat.numpy()), time.time()-start))


if __name__=="__main__":
    vocab = build_vocab(X)

    print("maxlength : ",max([len(s) for s in X]))
    print(vocab.onehot_sentence(train_X[0]).shape)

    for X,y in data_iter(32,train_X,train_Y):
        break
    print(vocab.onehot_sentences(X).shape)

    rnn = RNN(input_size=len(vocab))
    train_iter = data_iter(32, train_X, train_Y)
    test_iter = data_iter(32, test_X, test_Y)
    train_classify(rnn, vocab, train_iter, test_iter, 50, LR=0.01, num_epochs=250)

