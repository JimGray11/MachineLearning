#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/24 22:59
# @Author  : ywendeng
# @Version :v1.0
import numpy as np
import matplotlib.pyplot as plt

"""
使用梯度下降算法求解logistic_regression目标函数的最优解
"""


def load_date_set():
    data_mat = []
    label_mat = []
    with open("dataset/test_logistic/data") as file:
        for line in file:
            data = line.strip().split()
            data_mat.append([1.0, float(data[0]), float(data[1])])
            label_mat.append(int(data[2]))
    return data_mat, label_mat


def sigmoid(f):
    return 1.0 / (1 + np.exp(-f))


"""
伪代码：
    初始化回归系数为1
    重复下面步骤直到收敛{
    计算整个数据集的梯度
    使用alpha x gradient来更新回归系数
    }
    返回回归系数值
"""


def batch_gradient_descent(data, label):
    data_matrix = np.mat(data)
    # transpose 实现矩阵的转置[0,2,4,5](1,4)转置为(4,1)
    label_matrix = np.mat(label).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    n_epoch = 500  # 将整合训练集重复训练500
    # ones 表示产生10行1列的矩阵
    weights = np.ones((n, 1))
    for i in range(n_epoch):
        """
         批量梯度下降算法，在每次更新系数的时候，都不得不遍历整个数据集来计算整个数据集的误差
        """
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights += alpha * data_matrix.transpose() * error
    # 得到最优的参数解
    return weights


"""
伪代码：
    初始化回归系数为1
    重复下面步骤直到收敛{
        对数据集中每个样本
        计算该样本的梯度
        使用alpha x gradient来更新回归系数
    }
    返回回归系数值
"""


def stochastic_gradient_descent(data, label):
    m, n = np.shape(data)
    alpha = 0.1
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(data[i] * weights))
        error = label[i] - h
        weights += [alpha * error * k for k in data[i]]
    return weights


"""
 随机梯度下降算法过程中,系数有些时候会由于一些噪声(不正确的分类样本点)导致较大的波动。如果期望算法能够避免这种来回波动，从而收敛到每个值
 且收敛的速度也能加快。因此对随机梯度下降算法提出如下改进：
    1.每次迭代的时候，alpha都会更新，再具体点，都会变小，这回缓解数据的波动，或者高频波动。为了防止alpha太接近于0，会给alpha加一个常数项。
 这样做，可以保证多次迭代后新数据任然具有一定的影响，否则，再迭代就没啥意义了。
    2.每次迭代，通过随机选取样本来更新回归系数，这样可以减少周期性波动，因为样本顺序变了，使每次迭代不具备周期性。
"""


def optimization_stochastic_gradient_descent(data, label, n_epoch=150):
    m, n = np.shape(data)
    weights = np.ones(n)

    for j in range(n_epoch):
        index = []
        for k in range(m):
            index.append(k)
        for i in range(m):
            # alpha 每次都会减小，缓解数据的高频波动
            alpha = 4.0 / (1.0 + j + i) + 0.01
            # 每次随机选取样本来更新系数,这样可以减少周期性的波动
            rand_index = int(np.random.uniform(0, len(index)))
            h = sigmoid(np.sum(data[rand_index] * weights))
            error = label[rand_index] - h
            weights += [alpha * error * x for x in data[rand_index]]
            print "-->", rand_index
            index.remove(index[rand_index])
    return weights


def plot_best_fit(theta):
    data, label = load_date_set()
    data_arr = np.array(data)
    # 选取（100L,3L）中的第一数
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    """
     根据 theta[0]+ theta[1]*x1+theta[2]*x2=0 计算出x1,x2之间的关系
    """
    x = np.arange(-3.0, 3.0, 0.001)

    y = (-theta[0] - theta[1] * x) / theta[2]

    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


"""
对于计算分类正确率的方法:
def classify(testDir, weigh):
    dataArray, labelArray = loadData(testDir)
    dataMat = mat(dataArray)
    labelMat = mat(labelArray)
    h = sigmoid(dataMat * weigh)
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print int(labelMat[i]), 'is classified as : 1'
            if int(labelMat[i]) != 1:
                error += 1
                print 'error'
        else:
            print int(labelMat[i]), 'is classified as : 0'
            if int(labelMat[i]) != 0:
                error += 1
                print 'error'
    print 'error rate is ', '%.4f' %(error/m)
"""

if __name__ == "__main__":
    data, label = load_date_set()
    theta = optimization_stochastic_gradient_descent(data, label)
    plot_best_fit(theta)
