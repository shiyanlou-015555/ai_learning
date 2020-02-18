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
    建立语料库的符号字典。
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

    for i in range(0, num_examples, batch_size):
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