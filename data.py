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