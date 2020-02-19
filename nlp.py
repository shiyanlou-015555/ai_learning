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
import collections
import time

import rnn

def tokenize(sentences, token='word'):
    """
    对文章进行分词，其中文章由句子组成，其实句子分成单词或者字符。
    :param sentences: 文章句子组成的列表。
    :param token: 分词模式，默认按单词分，比标点符号也算单词。
    :return: 列表的列表，分词列表组成句子，句子组成文章。
    """
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)


def count_corpus(sentences):
    """
    建立符号集。
    :param sentences: 分好词的句子组成的列表。
    :return: 符号集合。
    """
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)


class Vocab(object):
    """
    建立语料库的符号字典，字符级别。
    * self.idx_to_token：列表，编号 -> 符号。
    * self.token_to_idx：字典，符号 -> 编号。
    * len(vocab)
    * vocab[token]、vocab[token_list]
    * vocab.to_tokens(idx)、vocab.to_tokens(idx_list)
    """
    def __init__(self, sentences, min_freq=0, use_special_tokens=False):
        """
        语料库字典初始化。
        :param tokens: 句子集合。
        :param min_freq: 分词超过该数量才建立索引。
        :param use_special_tokens: 使用特殊含义符号，如空格、句子开始、句子结束，否则只有未知符号。
        """
        counter = count_corpus(sentences)
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


class Vocab_dict(object):
    """
    建立字典的语料库，单词级别。
    """
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = collections.Counter(tokens)
        token_freqs = sorted(counter.items(), key=lambda x:x[0])
        token_freqs.sort(key=(lambda x:x[1]), reverse=True)
        if use_special_tokens:
            self.pad, self.bos, self.eos, self.unk = (0,1,2,3)
            tokens = ["<pad>","<bos>","<eos>","<unk>"]
        else:
            self.unk = 0
            tokens = ["<unk>"]
        tokens += [token for token, freq in token_freqs if freq >= min_freq]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, items):
        if not isinstance(items, (list, tuple)):
            return self.token_to_idx.get(items, self.unk)
        else:
            return [self.__getitem__(item) for item in items]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def build_vocab(tokens):
    """
    从字典建立语料库。
    """
    tokens = [token for line in tokens for token in line]
    return Vocab_dict(tokens, min_freq=3, use_special_tokens=True)


def pad(line, max_len, padding_token):
    """
    句子截断&补全。
    :param line: 分好词的句子。
    :param max_len: 最大长度。
    :param padding_token: 空白字符。
    :return: (step_nums, vocab_size)
    """
    if len(line) > max_len:
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line))


def build_array(lines, vocab, max_len, is_source):
    """
    建立字典的每一行从token转换为idx。
    array (batch_size, step_nums, vocab_size)
    :param lines: 字典。
    :param vocab: 字典语料库。
    :param max_len: 句子最大长度。
    :param is_source: 是否标注起始、结束。
    :return: array, valid_len
    """
    lines = [vocab[line] for line in lines]
    if not is_source:
        lines = [[vocab.bos] + line + [vocab.eos] for line in lines]
    array = torch.tensor([pad(line, max_len, vocab.pad) for line in lines])
    valid_len = (array != vocab.pad).sum(dim=1)
    return array, valid_len


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    """
    时序数据随机采样，样本不重叠。

    :param corpus_indices: 时序数据列表。
    :param batch_size: 批量大小，
    :param num_steps: 采样的时间步数。
    :param device: CUDA / CPU。
    :return: 迭代器（tensorX, tensorY）。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_examples = (len(corpus_indices) - 1) // num_steps
    example_indices = [i * num_steps for i in range(num_examples)]
    random.shuffle(example_indices)

    def _data(i):
        return corpus_indices[i: i + num_steps]

    for i in range(0, num_examples // batch_size * batch_size, batch_size):
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    """
    时序数据随机采样，样本不重叠，一批内数据间隔相等，相邻的两个随机小批量在原始序列上的位置相毗邻。

    :param corpus_indices: 时序数据列表。
    :param batch_size: 批量大小，
    :param num_steps: 采样的时间步数。
    :param device: CUDA / CPU。
    :return: 迭代器（tensorX, tensorY）。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size
    corpus_indices = corpus_indices[: corpus_len]
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


class Encoder(torch.nn.Module):
    """Encoder-Decoder的Encoder"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(torch.nn.Module):
    """Encoder-Decoder的Decoder"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """
    Encoder的具体实现。
    X shape: (batch_size, seq_len, embed_size)；
    Y shape: (seq_len, batch_size, num_hiddens)；
    LSTM的state包含最后一步的隐藏状态、记忆细胞，shape是 (num_layers, batch_size, num_hiddens)。
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout)

    def begin_state(self, batch_size, device):
        return [torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens), device=device)]

    def forward(self, X, *args):
        X = self.embedding(X)  # X shape: (batch_size, seq_len, embed_size)
        X = X.transpose(0, 1)  # RNN needs first axes to be time
        # state = self.begin_state(X.shape[1], device=X.device)
        out, state = self.rnn(X)
        return out, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.LSTM(embed_size,num_hiddens, num_layers, dropout=dropout)
        self.dense = torch.nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).transpose(0, 1)
        out, state = self.rnn(X, state)
        # Make the batch to be the first dimension to simplify loss computation.
        out = self.dense(out).transpose(0, 1)
        return out, state


def SequenceMask(X, X_len,value=0):
    """
    对于不定长序列求损失函数，仅在有效长度进行。
    :param X: (batch_size, maxlen)每一行是一个序列数据。
    :param X_len: (batch_size,)有效长度。
    :param value: mask。
    :return:
    """
    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :].to(X_len.device) < X_len[:, None]
    X[~mask]=value
    return X


class MaskedSoftmaxCELoss(torch.nn.CrossEntropyLoss):
    """
    带掩码的交叉熵损失函数类。

    X shape: (batch_size, seq_len, vocab_size)
    y shape: (batch_size, seq_len)
    valid_length shape: (batch_size, )
    """
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length).float()
        self.reduction='none'
        output=super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output*weights).mean(dim=1)


def train_ch7(model, data_iter, lr, num_epochs, device):
    """
    机器翻译的enc-dec训练。
    :param model: encoder-decode模型。
    :param data_iter: 数据迭代器（X, X_vlen, Y, Y_vlen）。
    :param lr: 学习率。
    :param num_epochs: 训练周期。
    :param device: CPU/GPU。
    :return: None
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    tic = time.time()
    for epoch in range(1, num_epochs + 1):
        l_sum, num_tokens_sum = 0.0, 0.0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_vlen, Y, Y_vlen = [x.to(device) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen - 1
            Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)

            l = loss(Y_hat, Y_label, Y_vlen).sum()
            l.backward()
            with torch.no_grad():
                rnn.grad_clipping_nn(model.parameters(), 5, device)
            num_tokens = Y_vlen.sum().item()
            optimizer.step()

            l_sum += l.sum().item()
            num_tokens_sum += num_tokens
        if epoch % 50 == 0:
            print("epoch {0:4d},loss {1:.3f}, time {2:.1f} sec".format(
                epoch, (l_sum / num_tokens_sum), time.time() - tic))
            tic = time.time()


def translate_ch7(model, src_sentence, src_vocab, tgt_vocab, max_len, device):
    """
    机器翻译的enc-dec预测。
    :param model: encoder-decode模型。
    :param src_sentence: 待翻译语句。
    :param src_vocab: 源语言词典。
    :param tgt_vocab: 目标语言词典。
    :param max_len: 最大有效句子长度。
    :param device: CPU/GPU。
    :return: 翻译好的语句。
    """
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    src_len = len(src_tokens)
    if src_len < max_len:
        src_tokens += [src_vocab.pad] * (max_len - src_len)
    enc_X = torch.tensor(src_tokens, device=device)
    enc_valid_length = torch.tensor([src_len], device=device)
    enc_outputs = model.encoder(enc_X.unsqueeze(dim=0), enc_valid_length)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_length)
    dec_X = torch.tensor([tgt_vocab.bos], device=device).unsqueeze(dim=0)
    predict_tokens = []
    for _ in range(max_len):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # The token with highest score is used as the next time step input.
        dec_X = Y.argmax(dim=2)
        py = dec_X.squeeze(dim=0).int().item()
        if py == tgt_vocab.eos:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))