#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/20 21:26
# @Author  : ywendeng
# @Version : v1.0
"""
  线性回归的使用批量梯度下降算法求解
  学习资源：https://python.freelycode.com/contribution/detail/577
  正则化：http://www.cnblogs.com/jianxinzhou/p/4083921.html
  1.首先加载数据集，将字符串值转换为数字，然后将每个列统一化为0到1的之间的值.
  这里由辅助函数load_csv()和str_column_to_float()加载和初步处理数据集，
  并由dataset_minmax()和normalize_dataset()来统一化
  2.使用k-折交叉验证评估模型对未知数据的性能。
  这意味着，我们将建模并评估k模型，估算的平均模型误差下的性能。将使用标准差来评估每个模型。
  这些功能由cross_validation_split()，rmse_metric()和evaluate_algorithm()辅助函数中提供。
  备注：k-折交叉验证：在机器学习中，将数据集A分为训练集（training set）B和测试集（test set）C，
  在样本量不充足的情况下，为了充分利用数据集对算法效果进行测试，
  将数据集A随机分为k个包，每次将其中一个包作为测试集，剩下k-1个包作为训练集进行训练。
"""
from csv import reader
from math import sqrt
from random import randrange
from random import seed


# 加载数据文件
def load_csv(file_name):
    data_set = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_set.append(row)
    return data_set


# 转换string类型为float类型
def str_column_to_float(data_set, column):
    for row in data_set:
        row[column] = float(row[column].strip())


# 为每列找到最小和最大值
def dataset_minmax(dataset):
    min_max = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        min_max.append([value_min, value_max])

    return min_max


# 调整列数据范围为0-1 之间
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# 拆分数据集为k折
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            # list 中的pop 是删除特定下标的元素，remove 是删除list中的指定元素，而不是下标索引
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# 计算标准差
def rmse_matric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# 使用交叉验证来评估一个算法
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        # 获取训练集
        train_set = list(folds)
        train_set.remove(fold)
        # 将训练集转换为一个list [[1,3],[5,2]] 转变之后变成[1,3,5,2]
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_matric(actual, predicted)
        scores.append(rmse)
    return scores


# 用系数进行预测-----预测函数可能在两个地方使用1.在评估随机梯度下降算法的候选参数 2.在建模完成之后用于预测数据
def predict(row, coefficients):
    # 第一个系数总是截距，也被称为偏置或b0，因为它是独立的，对任何一个特定的输入值没什么关联
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


# 使用随机梯度下降算法来算系数
"""
随机梯度下降法需要两个参数：
学习率：用于每次更新时每次校正时，限制系数调整的大小。
Epochs：即在更新系数时运行训练数据的次数。（译注:1个epoch等于使用训练数据集中的全部样本训练一次，可理解为一个训练周期）
"""


def coefficient_sgd(train, l_rate, n_epoch):
    # 将系数全部初始化为0
    coef = [0.0 for i in range(len(train[0]))]
    lost_function = list()
    for epoch in range(n_epoch):
        # 用于记录运行一次训练数据的损失函数的方差
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            # 记录一下每使用一次训练数据的误差的平方差
            sum_error += error ** 2
            # 更新常系数项
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        lost_function.append(sum_error)
        print(">epoch=%d,lrate=%.3f,error=%.3f" % (epoch, l_rate, sum_error))
    return coef


# 利用随机梯度下降算法估算线性模型参数
def liner_regression_sgd(train_set, test_set, l_rate, n_epoch):
    predictions = list()
    coef = coefficient_sgd(train_set, l_rate, n_epoch)
    for row in test_set:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return predictions


if __name__ == "__main__":
    # 葡萄酒质量数据集的线性回归
    seed(1)  # seed() 方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数
    # 载入准备数据
    file_name = u"dataset/winequality-white.csv"
    dataset = load_csv(file_name)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # 归一化
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # 评估算法
    n_folds = 5
    l_rate = 0.01
    n_epoch = 50
    scores = evaluate_algorithm(dataset, liner_regression_sgd, n_folds,l_rate, n_epoch)
    print "scores:%s" % scores
    print "Mean RMSE:%.3f" % (sum(scores) / float(len(scores)))
