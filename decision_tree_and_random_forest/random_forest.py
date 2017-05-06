#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 19:21
# @Author  : ywendeng
# @Version : v1.0
"""
 随机森林的算法思想：
   随机森林就是集成(ensemble learning)学习的思想将多棵树集成一种算法,它的基本思想是决策树，每棵决策树都是一个分类器
 ，那么对于一个输入样本，N棵树会有N个分类结果。而随机森林集成了所有分类投票结果，将投票次数最多的类别指定为最终的输出——即最简单的Bagging
 思想
 随机森林中每棵树的生成规则：
     1）如果训练集大小为N，对于每棵树而言，随机且有放回地从训练集中的抽取M个训练样本（这种采样方式称为bootstrap sample方法），作为该树的训练集
     2）如果每个样本的特征维度为M，指定一个常数m<<M，随机地从M个特征中选取m个特征子集，每次树进行分裂时，从这m个特征中选择最优的

 ---随机森林就是对决策树的集成，但有两点不同：
     1）采样的差异性：从含m个样本的数据集中有放回的采样，得到含m个样本的采样集，用于训练。这样能保证每个决策树的训练样本不完全一样。
     2）特征选取的差异性：每个决策树的n个分类特征是在所有特征中随机选择的（n是一个需要我们自己调整的参数）
 ---随机森林需要调整的参数有：
     1）决策树的个数
     2）特征属性的个数
     3）递归次数（即决策树的深度）
 随机森林算法实现：
    （1）    导入文件并将所有特征转换为float形式
    （2）    将数据集分成n份，方便交叉验证
    （3）    构造数据子集（随机采样），并在指定特征个数（假设m个，手动调参）下选取最优特征
    （4）    构造决策树
    （5）    创建随机森林（多个决策树的结合）
    （6）    输入测试集并进行测试，输出预测结果
"""

import csv
from random import randrange
from random import seed

"""
 加载是数据集
"""


# 数据加载---使用csv来读取csv 文件格式的数据
def load_csv(file_name):
    data_set = []
    with open(file_name, 'r')as file:
        reader = csv.reader(file)
        for line in reader:
            data_set.append(line)
    return data_set


# 除了判别列之外，其他列都转换为float类型
def column_to_float(data):
    feature_len = len(data[0]) - 1
    for line in data:
        for i in range(feature_len):
            line[i] = float(line[i].strip())


"""
 将数据集分成N份,方便交叉验证
"""


def split_data_set(dataset, n_folds):
    fold_size = int(len(dataset) / n_folds)
    data_copy = list(dataset)
    data_split = []
    for i in range(n_folds):
        fold_data = []
        while len(fold_data) < fold_size:
            index = randrange(len(data_copy))
            fold_data.append(data_copy.pop(index))
        data_split.append(fold_data)
    return data_split


"""
 采用随机采样的方法构造数据子集,和随机选取特征值
"""


# 构造数据子集
def get_sub_sample(dataset, ratio):
    sub_dataset = []
    # 从m 个样本中随机选择一个样本子集
    sub_data_len = round(len(dataset) * ratio)  # 返回浮点数x的四舍五入值。
    while len(sub_dataset) < sub_data_len:
        index = randrange(len(dataset) - 1)
        sub_dataset.append(dataset[index])
    return sub_dataset


# 分割数据集
def data_spilt(dataSet, index, value):
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 计算分割代价
def spilt_loss(left, right, class_values):
    loss = 0.0
    for class_value in class_values:
        left_size = len(left)
        if left_size != 0:  # 防止除数为零
            prop = [row[-1] for row in left].count(class_value) / float(left_size)
            loss += (prop * (1.0 - prop))
        right_size = len(right)
        if right_size != 0:
            prop = [row[-1] for row in right].count(class_value) / float(right_size)
            loss += (prop * (1.0 - prop))
    return loss


# 选取n个特征，在n个特征中，选取分割时的最优特征
def get_best_spilt(dataset, n_features):
    features = []
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    """
    训练部分：假设我们取dataset中的m个feature来构造决策树，首先，我们遍历m个feature中的每一个feature，
    再遍历每一行，通过spilt_loss函数（计算分割代价）来选取最优的特征及特征值，根据是否大于这个特征值进行分类（分成left,right两类），
    循环执行上述步骤，直至不可分或者达到递归限值（用来防止过拟合），最后得到一个决策树tree。
    """
    for index in features:
        for row in dataset:

            left, right = data_spilt(dataset, index, row[index])
            loss = spilt_loss(left, right, class_values)
            if loss < b_loss:
                b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right
    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}


# 决定输出标签
def decide_label(data):
    output = [row[-1] for row in data]
    return max(set(output), key=output.count)


# 子分割，不断地构建叶节点的过程
def sub_spilt(root, n_features, max_depth, min_size, depth):
    left = root['left']
    # print left
    right = root['right']
    del (root['left'])
    del (root['right'])
    # print depth
    if not left or not right:
        root['left'] = root['right'] = decide_label(left + right)
        # print 'testing'
        return
    if depth > max_depth:
        root['left'] = decide_label(left)
        root['right'] = decide_label(right)
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        root['left'] = get_best_spilt(left, n_features)
        # print 'testing_left'
        sub_spilt(root['left'], n_features, max_depth, min_size, depth + 1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        root['right'] = get_best_spilt(right, n_features)
        # print 'testing_right'
        sub_spilt(root['right'], n_features, max_depth, min_size, depth + 1)

        # 构造决策树


def build_tree(dataSet, n_features, max_depth, min_size):
    root = get_best_spilt(dataSet, n_features)
    sub_spilt(root, n_features, max_depth, min_size, 1)
    return root


# 预测测试集结果
def predict(tree, row):
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# 创建随机森林
def random_forest(train, test, ratio, n_feature, max_depth, min_size, n_trees):
    trees = []
    for i in range(n_trees):
        train = get_sub_sample(train, ratio)
        tree = build_tree(train, n_feature, max_depth, min_size)
        # print 'tree %d: '%i,tree
        trees.append(tree)
    # predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]
    return predict_values


# 计算准确率
def accuracy(predict_values, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predict_values[i]:
            correct += 1
    return correct / float(len(actual))


if __name__ == '__main__':
    seed(1)
    dataSet = load_csv('./dataset/sonar.all-data.csv')
    column_to_float(dataSet)
    n_folds = 5
    max_depth = 10
    min_size = 1
    ratio = 1.0
    # n_features=sqrt(len(dataSet)-1)
    n_features = 15
    n_trees = 10
    folds = split_data_set(dataSet, n_folds)
    scores = []
    for fold in folds:
        # 此处不能简单地用train_set=folds，这样用属于引用,那么当train_set的值改变的时候，
        # folds的值也会改变，所以要用复制的形式。（L[:]）能够复制序列，D.copy() 能够复制字典，list能够生成拷贝 list(L)
        train_set = folds[:]
        train_set.remove(fold)
        # print len(folds)
        train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表
        # print len(train_set)
        test_set = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
            # for row in test_set:
            # print row[-1]
        actual = [row[-1] for row in fold]
        predict_values = random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)
        accur = accuracy(predict_values, actual)
        scores.append(accur)
    print ('Trees is %d' % n_trees)
    print ('scores:%s' % scores)
    print ('mean score:%s' % (sum(scores) / float(len(scores))))
