# coding: utf-8

import torch
import torchvision
from torch.nn import init
import torch.utils.data as data
# print(torch.__version__)
# print(torchvision.__version__)
# print(torch.cuda.is_available())
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
import re
import zipfile


mnist_train = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST',
                                                train=True,
                                                download=True,
                                                transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST',
                                               train=False,
                                               download=True,
                                               transform=torchvision.transforms.ToTensor())


def get_fashion_mnist_labels(labels):
    """
    FASHION_MNIST数据集的数字标签转文本标签。

    :param labels:数字标签数组，标签取值范围0~9；
    :return:文本标签数组。
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """
    批量显示FASHION_MNIST数据集的图像。
    :param images:图像数组，每个图像是一个（Channel=1 x Height x Width）的Tensor。
    :param labels:文本标签数组。
    :return:None
    """
    display.set_matplotlib_formats("svg")
    fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
    for ax, img, lbl in zip(axes, images, labels):
        ax.imshow(img.view((28, 28)).numpy(), cmap='gray_r')
        ax.set_title(lbl)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


def read_time_machine():
    """
    获取小说《timemachine》
    :return: timechine各行文本组成的列表。
    """
    with open('Datasets/timemachine.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines


def load_data_jay_lyrics():
    """
    获取周杰伦歌词数据集。

    :return: corpus_indices, char_to_idx(字典), idx_to_char(列表), vocab_size
    """
    with zipfile.ZipFile('DataSets/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size