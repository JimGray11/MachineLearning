#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/23 9:48
# @Author  : ywendeng
# @Version : v1.0
"""
 主要使用logistic_regression 来实现数据集的分类问题
"""

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# 分类标签key-value键值对
def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = "dataset/iris/4.iris.data"
    # 加载数据 -----路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
    data = np.loadtxt(path, dtype=float, delimiter=",", converters={4: iris_type})
    # 将数据0到3列组成x, 第4列得到y
    x, y = np.split(data, (4,), axis=1)
    # 为了可视化方便,仅使用前两列特征
    x = x[:, :2]
    logreg = LogisticRegression()
    # y.ravel() 主要是为使得[[1],[2]] 变成[1,2]的形式
    logreg.fit(x, y.ravel())

    # -----------------------------------------------
    # 画图
    N, M = 1000,1000  # 横纵各采样多少个值
    # 找到第0列的最大值和最小值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    # 找到第1列的最大值和最小值
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)

    # 生成网格采样点----np.meshgrid()的作用
    """
    x = 1:3;
    y = 10:14;
    [X, Y] = meshgrid(x, y);
    其结果为：
    X =
    1     2     3
    1     2     3
    1     2     3
    1     2     3
    1     2     3
    Y =
    10    10    10
    11    11    11
    12    12    12
    13    13    13
    14    14    14
    """
    x1, x2 = np.meshgrid(t1, t2)
    """
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    >>> np.stack((a, b))
        array([[1, 2, 3],
         [2, 3, 4]])
     x1--->[ 4.3         4.30721443  4.31442886 ...,  7.88557114  7.89278557  7.9       ]
     x2--->[ 2.          2.          2.         ...,  2.          2.          2.        ]
    """
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    # -------------------------------------------------
    y_hat = logreg.predict(x_test)
    # 为了使得输出相同，则需要将y_hat 重新变形
    y_hat = y_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_hat, cmap=plt.cm.Spectral,
                   alpha=0.1)  # 预测值的显示Paired/Spectral/coolwarm/summer/spring/OrRd/Oranges
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.prism)  # 样本的显示
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()

    # 训练集上的预测结果
    y_hat = logreg.predict(x)
    y = y.reshape(-1)  # 此转置仅仅为了print时能够集中显示
    result = (y_hat == y)  # True则预测正确，False则预测错误
    print y_hat
    print y
    print result
    c = np.count_nonzero(result)  # 统计预测正确的个数
    print c
    print 'Accuracy: %.2f%%' % (100 * float(c) / float(len(result)))
    """"
    备注:meshgrid ,stack,flat的测试
    h = np.linspace(1, 2, 5)
    t = np.linspace(3, 4, 5)
    h1, t1 = np.meshgrid(h, t)
    print np.stack((h1.flat,t1.flat),axis=1)
    """
