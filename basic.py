# coding: utf-8

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


def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def data_iter(batch_size, features, labels):
    """
    遍历数据集并不断读取小批量数据样本。

    :param batch_size:批量大小；
    :param features:特征Tensor；
    :param labels:标签Tensor；
    :return:小样本（特征,标签）迭代器；
    """
    num_examples = len(features)
    indices = list(range(0,num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j), labels.index_select(0,j)


class FlattenLayer(torch.nn.Module):
    """
    把高维度特征展开成一维特征，每个示例一行。
    (batch_size, c1, c2, c3, ...) -> (barch_size, features_num=c1*c2*c3*...)
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


def linreg(X, W, b):
    """
    线性回归。

    :param X:输入矩阵（batch_size, features_num）；
    :param W:权重矩阵（features_num, lables_num)，labels_num一般为1；
    :param b:偏置向量（lables_num)；
    :return:输出矩阵(batch_size, labels_num)；
    """
    return X.mm(W)+b


def softmax(X):
    """
    转换后，任意一行元素代表了一个样本在各个输出类别上的预测概率，概率之和为1。

    :param X:特征强度矩阵（batch_size, features_num）；
    :return:特赠概率矩阵（batch_size, features_num）；
    """
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def relu(O):
    """
    对隐藏层的输出进行激活max(0,o)。
    :param O: 上一层的线性变换结果(batch_size, hiddens_num)
    :return: 隐藏层的输出(batch_size, hiddens_num)
    """
    return torch.max(input=O, other=torch.tensor(0.0))


def sigmoid(O):
    """
    对隐藏层的输出进行sigmoid激活。
    :param O: 上一层的线性变换结果(batch_size, hiddens_num)
    :return: 隐藏层的输出(batch_size, hiddens_num)
    """
    return - 1 / (1+O.exp())


def tanh(O):
    """
        对隐藏层的输出进行tanh激活。
        :param O: 上一层的线性变换结果(batch_size, hiddens_num)
        :return: 隐藏层的输出(batch_size, hiddens_num)
        """
    return (1-torch.exp(-2*0))/(1+torch.exp(-2*0))


def squared_loss(y_hat, y):
    """
    计算平方损失函数，一般y_hat为（batch_size,1）的二维矩阵，y为(batch_size,)的一维向量。

    :param y_hat:模型预测值
    :param y:标签真实值
    :return:损失矩阵（batch_size,1），每个示例的平方损失。
    """
    return (y_hat-y.reshape(y_hat.shape)) ** 2 / 2


def cross_entropy(y_hat, y):
    """
    求预测概率矩阵和真实标签onehot的交叉熵。

    :param y_hat:各个类别上的概率矩阵（batch_size, labels_num)
    :param y:真实标签向量(batch_size，)
    :return:各个示例的交叉熵（batch_size,1)
    """
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def l2_penalty(w):
    """
    计算模型的l2惩罚项。

    :param w:模型的权重；
    :return: 模型的L2惩罚项，tensor.shape=(1,)；
    """
    return (w**2).sum() / 2


def accuracy(y_hat, y):
    """
    求预测概的准确率(TP+TN)/(TP+TN+FP+FN)。

    :param y_hat:各个类别上的概率矩阵（batch_size, labels_num)
    :param y:真实标签向量(batch_size，)
    :return: 准确度指标
    """
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(dataiter, net, device=None):
    """
    求模型在数据集上的准确率指标，评估时不遗忘。

    :param dataiter: 数据集迭代器（features, labels)
    :param net: 模型。
    :param device: CPU/CUDA。
    :return: 准确度指标
    """
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        # 模型预测模式不需要反向传播。
        for X, y in dataiter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train()
            else:
                if ("is_training" in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().cpu().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降优化器。

    :param params:待优化模型的参数。
    :param lr:学习率。
    :param batch_size:批量大小。
    :return:
    """
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    """
    训练深度学习分类模型（网络）。
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, net.parameters(), lr)

    :param net:深度学习模型。
    :param train_iter:训练数据集迭代器（features, labels）。
    :param test_iter:测试数据集迭代器（features, labels）。
    :param loss:损失函数。
    :param num_epochs:学习周期。
    :param batch_size:批量大小。
    :param params:模型参数集。
    :param lr:学习率。
    :param optimizer:优化器。
    :return:None
    """
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is not None:
                optimizer.step()
            else:
                sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    """
    训练深度学习分类模型，损失函数为交叉熵函数。

    :param net: 模型。
    :param train_iter: 训练数据集迭代器（features, labels）。
    :param test_iter: 测试数据集迭代器（features, labels）。
    :param batch_size: 批量大小。
    :param optimizer: 优化器。
    :param device: CPU/CUDA。
    :param num_epochs: 训练周期。
    :return: None。
    """
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    """
    x_vals, y_vals, x_label, y_labe可迭代。
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)




from data import *

if __name__ == "__main__":

    batch_size = 256
    num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_epochs, lr = 100, 0.3
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)
    params1 = [W, b]
    for param in params1:
        param.requires_grad_(requires_grad=True)
        param.to(torch.device("cuda"))
    def net_sigmoid(X):
        num_inputs = W.shape[0]
        return softmax(torch.mm(X.view(-1, num_inputs), W) + b)
    loss = cross_entropy
    trainloss_sigmoid = []
    testloss_sigmoid = []
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net_sigmoid(X)
            l = loss(y_hat, y).sum()
            l.backward()
            sgd([W, b], lr, batch_size)
            for param in params1:
                param.grad.data.zero_()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net_sigmoid)
        trainloss_sigmoid.append(train_acc_sum / n)
        testloss_sigmoid.append(test_acc)
        print("epoch {:d}, train_loss {:.4f}, train_acc_sum {:.3f}, test_acc {:.3f}".format(epoch, train_l_sum / n,
                                                                                            train_acc_sum / n,
                                                                                            test_acc))

    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)
    params2 = [W1, b1, W2, b2]
    for param in params2:
        param.requires_grad_(requires_grad=True)
        param.to(torch.device("cuda"))
    def net_mlp(X):
        X = X.view((-1, num_inputs))
        H = relu(torch.matmul(X, W1) + b1)
        return torch.matmul(H, W2) + b2
    loss = torch.nn.CrossEntropyLoss()
    optimizer2 = torch.optim.SGD(params2, lr)
    trainloss_mlp = []
    testloss_mlp = []
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net_mlp(X)
            l = loss(y_hat, y).sum()
            l.backward()
            optimizer2.step()
            for param in params2:
                param.grad.data.zero_()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net_mlp)
        trainloss_mlp.append(train_acc_sum / n)
        testloss_mlp.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
        epoch, train_l_sum / n, train_acc_sum / n, test_acc))

    import pickle
    store = {"trainloss_sigmoid":trainloss_sigmoid,
             "testloss_sigmoid":testloss_sigmoid,
             "trainloss_mlp":trainloss_mlp,
             "testloss_mlp":testloss_mlp}
    with open("temp/lr_"+str(lr)+".pkl", "wb") as f:
        pickle.dump(store,f)
