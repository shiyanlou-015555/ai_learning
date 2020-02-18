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


def corr2d(X, K):
    """
    对二维特折X与核K进行卷积运算。

    :param X:二维输入特征（H,W）
    :param K:核数组（h,w）
    :return:二维输出。
    """
    h, w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y


class Conv2D(torch.nn.Module):
    """
    构造二维卷积层
    """
    def __init__(self, kernel_size):
        """
        构造二维卷积层
        :param kernel_size: 核大小(h,w)。
        """
        super(Conv2D, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, X):
        """
        前向运算，
        :param X: 二维输入特征（H,W）。
        :return: 二维输出。
        """
        return corr2d(X, self.weight) + self.bias


def corr2d_multi_in(X, K):
    """
    多输入通道的二维互相关运算。

    :param X: 三维输入特征（in_channels, Height, Width）
    :param K: 三维卷积数组(in_channels, height, width)
    :return: 二维输出，与输入通道数无关。
    """
    Y = corr2d(X[0,:,:], K[0,:,:])
    for i in range(1,X.shape[0]):
        Y += corr2d(X[i,:,:], K[i,:,:])
    return Y


def corr2d_multi_in_out(X, K):
    """
    多输入&输出通道的二维互相关运算。

    :param X: 三维输入特征（in_channels, Height, Width）
    :param K: 四维卷积数组(out_channels, in_channels, height, width)
    :return: 三维输出，与输入通道数无关(out_channels, height, width)。
    """
    return torch.stack([corr2d_multi_in(X,k) for k in K],dim=0)


def corr2d_multi_in_out_1x1(X, K):
    """
    1x1卷积层等效于通道维的全连接层。

    :param X: 三维输入特征（in_channels, Height, Width）
    :param K: 四维卷积数组(out_channels, in_channels, 1, 1)
    :return: 三维输入特征（out_channels, Height, Width）
    """
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)


def pool2d(X, pool_size, mode='max'):
    """
    对二维特征X进行池化操作。
    :param X: 二维输入特征（H,W）
    :param pool_size: 池化窗口大小(p,q)。
    :param mode: 二维输出。
    :return:
    """
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


class Flatten(torch.nn.Module):
    """
    把高维特征展平成一维特征。
    """
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(torch.nn.Module):
    """
    规范LeNet的输入特征为1x28x28。
    """
    def forward(self, x):
        return x.view(-1,1,28,28)