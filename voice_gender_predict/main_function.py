#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/1/18 21:34
# @Author  : ywendeng
# @Version : v1.0
import pandas as pd
from pandas_tools import inspect_csv_data, process_missing_data, visualize_two_charecter, \
    visualize_multi_charecter, visualize_single_charecter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

"""
  数据特征已经从原始数据中提取好-----根据现有声音数据来预测性别
  0.明确分析目标
  1.查看数据
  2.处理缺失数据
  3.数据统计分析-------可视化特征分布
  4.选择模型
    ----模型训练
    -----交叉验证(可选)
  5.保存分析结果
    ----- 保存模型
    ----- 测试模型
"""


def run_main():
    file_path = "./dataset/voice.csv"
    df_data = pd.read_csv(file_path)
    # 1.查看数据中的基本信息
    inspect_csv_data(df_data)
    # 2.处理数据中的缺失值
    df_data = process_missing_data(df_data)
    # 可视化单个特征
    # visualize_single_charecter(df_data, 'meanfun')
    # 可视化特征值-----查看两个特征之间的关系
    # visualize_two_charecter(df_data, 'meanfun', 'centroid')
    # 可视化多个特征之间的关系
    fea_names = ['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']
    # visualize_multi_charecter(df_data, fea_names)
    # 为模型准备数据----------在数据集中相当于特征选择已经是选择了
    X = df_data.iloc[:, :-1].values
    df_data["label"].replace('male', 0, inplace=True)
    df_data["label"].replace('female', 1, inplace=True)
    y = df_data["label"].values

    # 对特征做归一化
    X = preprocessing.scale(X)
    # 使用sklearn中train_test_spilt 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3., random_state=5)
    # 选择模型----由于是监督学习中分类问题，则选择最为简单的knn聚类算法，并使用交叉验证，来训练参数
    k_range = range(1, 31)
    cross_score = []
    for k in k_range:
        knn = KNeighborsClassifier(k)
        score = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        score_mean = score.mean()
        cross_score.append(score_mean)
        print "%i: %.4f" % (k, score_mean)
    best_k = np.argmax(cross_score) + 1
    print "最优的K:", best_k
    # 对最优参数画图
    plt.plot(k_range, cross_score)
    plt.xlabel("Accuracy")
    plt.ylabel("K")
    plt.show()

    # 交叉验证得到最优的参数之后，需要进行模型训练
    knn_model = KNeighborsClassifier(best_k)
    knn_model.fit(X_train, y_train)
    # 使用模型预测数据
    print "预测精度为%.4f" % knn_model.score(X_test, y_test)

    return None


if __name__ == "__main__":
    run_main()
