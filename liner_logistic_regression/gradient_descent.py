#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/20 11:10
# @Author  : ywendeng
# @Version :  v1.0
"""
梯度下降算法是最优化的问题求解算法，有批量梯度下降算法和随机梯度下降算法两种不同的迭代思路，其区别如下：
1.批量梯度算法每次迭代都能够得到全局最优解，但是每迭代一步，都要用到训练集所有的数据，如果样本数目m很大，则开销很大
  优点：全局最优解；易于并行实现；
  缺点：当样本数目很多时，训练过程会很慢。
2.随机梯度下降是通过每个样本来迭代更新一次，如果样本量很大的情况（例如几十万），那么可能只用其中几万条或者几千条的样本，
  就已经将theta迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次。
  但是，SGD伴随的一个问题是噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。
　优点：训练速度快；
　缺点：准确度下降，并不是全局最优；不易于并行实现。
"""

# 随机梯度下降算法的实现-----每一次梯度下降只需要选取一个样本数据

"""
    :param x: 训练集种的自变量
    :param y: 训练集种的因变量
    :param theta: 待求的权值
    :param alpha: 学习速率
    :param m: 样本总数
    :param max_iter: 最大迭代次数----在每次迭代中只使用一个样本
    :param EPS 表示误差率
"""


def stochasticGradientDescent(x, y, theta, alpha, m, max_iter, EPS):
    # 偏差,误差
    deviation = 1
    iter = 0
    flag = 0
    while True:
        # 循环迭代训练集
        for i in range(m):
            deviation = 0
            h = theta[0] * x[i][0] + theta[1] * x[i][1]
            theta[0] = theta[0] + alpha * (y[i] - h) * x[i][0]
            theta[1] = theta[1] + alpha * (y[i] - h) * x[i][1]

            iter = iter + 1
            # 计算所有样本点的误差均值
            for i in range(m):
                deviation = deviation + (y[i] - (theta[0] * x[i][0] + theta[1] * x[i][1])) ** 2
            if deviation < EPS or iter > max_iter:
                flag = 1
                break
        if flag == 1:
            break

    return theta, iter


# 批量梯度下降算法---使用训练集中的所有样本
def batchGradientDescent(x, y, theta, alpha, m, max_iter, EPS):
    devition = 1
    iter = 0
    while devition > EPS and iter < max_iter:
        devition = 0
        sigma1 = 0
        sigma2 = 0
        # 对训练集中的所有数据求和迭代
        for i in range(m):
            h = theta[0] * x[i][0] + theta[1] * x[i][1]
            sigma1 = sigma1 + (y[i] - h) * x[i][0]
            sigma2 = sigma2 + (y[i] - h) * x[i][1]
        theta[0] = theta[0] + alpha * sigma1/m
        theta[1] = theta[1] + alpha * sigma2/m
        # 计算误差
        for i in range(m):
            devition = devition + (y[i] - (theta[0] * x[i][0] + theta[1] * x[i][1])) ** 2
        iter = iter + 1

    return theta, iter


def run_stochastic():
    EPS = 0.0001
    matrix_x = [[2.1, 1.5], [2.5, 2.3], [3.3, 3.9], [3.9, 5.1], [2.7, 2.7]]
    matrix_y = [2.5, 3.9, 6.7, 8.8, 4.6]
    max_iter = 5000
    # 随机梯度初始化
    theta = [2, -1]
    alpha = 0.05

    # 随机梯度计算之后的结果
    resultTheta, iters = stochasticGradientDescent(
        matrix_x, matrix_y, theta, alpha, 5, max_iter, EPS
    )

    print "theta=", resultTheta
    print "iters=", iters


def run_batch():
    EPS = 0.0001
    matrix_x = [[2.1, 1.5], [2.5, 2.3], [3.3, 3.9], [3.9, 5.1], [2.7, 2.7]]
    matrix_y = [2.5, 3.9, 6.7, 8.8, 4.6]
    max_iter = 5000
    # 随机梯度初始化
    theta = [2, -1]
    alpha = 0.05

    # 随机梯度计算之后的结果
    resultTheta, iters = batchGradientDescent(
        matrix_x, matrix_y, theta, alpha, 5, max_iter, EPS
    )

    print "theta=", resultTheta
    print "iters=", iters


if __name__ == "__main__":
    run_batch()
