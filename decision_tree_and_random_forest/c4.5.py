#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/3 10:24
# @Author  : ywendeng
# @Version : v1.0
from math import log
import operator
import treePlotter

"""
 决策树C4.5的实现
"""


def create_data_set():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    data_set = [[0, 0, 0, 0, 'N'],
                [0, 0, 0, 1, 'N'],
                [1, 0, 0, 0, 'Y'],
                [2, 1, 0, 0, 'Y'],
                [2, 2, 1, 0, 'Y'],
                [2, 2, 1, 1, 'N'],
                [1, 2, 1, 1, 'Y']]
    label = ['outlook', 'temperature', 'humidity', 'windy']
    return data_set, label


# 计算数据集的经验熵
def calculate_entropy(data):
    row_num = len(data)
    label_count = {}
    for row in data:
        label = row[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    entropy = 0.0
    for key in label_count:
        prob = float(label_count[key]) / row_num
        entropy -= prob * log(prob, 2)
    return entropy


# 子数据集的划分
def split_data_set(dataset, colume, value):
    sub_data_set = []
    for row in dataset:
        if row[colume] == value:
            reduce_future_value = row[:colume]
            reduce_future_value.extend(row[colume + 1:])  #
            sub_data_set.append(reduce_future_value)  # 根据信息增益率选择增益率最大的作为分裂结点
    return sub_data_set


def choose_best_feature_to_split(dataset):
    num_feature = len(dataset[0]) - 1  # feature的特征个数
    base_entropy = calculate_entropy(dataset)  # 计算整个数据集的经验熵
    best_gain_ratio = 0.0
    best_feature = -1
    # 循环遍历计算每个特征对数据集的信息增益率
    for i in range(num_feature):
        feature_list_value = [example[i] for example in dataset]  # 每个feature的list
        # 选择出每个feature的取值{a1,a2,a3......an},将数据集D划分为D1,D2.....Dn个子集
        unique_feature_value = set(feature_list_value)
        new_entropy = 0.0
        feature_entropy = 0.0
        # 根据不同的特征中的子数据集划
        for value in unique_feature_value:
            sub_data_set = split_data_set(dataset, i, value)
            sub_data_set_ratio = len(sub_data_set) / float(len(dataset))
            new_entropy += sub_data_set_ratio * calculate_entropy(sub_data_set)
            feature_entropy += -sub_data_set_ratio * log(sub_data_set_ratio, 2)
        gain = base_entropy - new_entropy
        if feature_entropy == 0:
            continue
        gain_ratio_info = gain / feature_entropy
        # 寻找信息增益率最大的feature
        if gain_ratio_info > best_gain_ratio:
            best_gain_ratio = gain_ratio_info
            best_feature = i
    return best_feature


def majority_count(label):
    label_count = {}
    for v in label:
        if v not in label_count.keys():
            label_count[v] = 0
        label_count[v] += 1
    # 表示根据计算的值进行来选取样本数量最多的标签
    sorted_label = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label[0][0]


# 多重字典构建树
def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    classList = [example[-1] for example in dataSet]  # ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y']
    if classList.count(classList[0]) == len(classList):
        # classList所有元素都相等，即类别完全相同，停止划分
        return classList[0]  # splitDataSet(dataSet, 0, 0)此时全是N，返回N
    if len(dataSet[0]) == 1:  # [0, 0, 0, 0, 'N']
        # 遍历完所有特征时返回出现次数最多的
        return majority_count(classList)
    bestFeat = choose_best_feature_to_split(dataSet)  # 0－> 2
    # 选择最大的gain ratio对应的feature
    bestFeatLabel = labels[bestFeat]  # outlook -> windy
    myTree = {bestFeatLabel: {}}
    # 多重字典构建树{'outlook': {0: 'N'
    del (labels[bestFeat])  # ['temperature', 'humidity', 'windy'] -> ['temperature', 'humidity']
    featValues = [example[bestFeat] for example in dataSet]  # [0, 0, 1, 2, 2, 2, 1]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # ['temperature', 'humidity', 'windy']
        myTree[bestFeatLabel][value] = createTree(split_data_set(dataSet, bestFeat, value), subLabels)
        # 划分数据，为下一层计算准备
    return myTree


def view_tree():
    dataSet, labels = create_data_set()
    labels_tmp = labels[:]
    desicionTree = createTree(dataSet, labels_tmp)
    treePlotter.createPlot(desicionTree)

    # 遍历完成所有特征时返回出现次数最多的


if __name__ == '__main__':
    view_tree()
