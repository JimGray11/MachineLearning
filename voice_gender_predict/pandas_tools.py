#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/18 21:34
# @Author  : ywendeng
# @Version : v1.0
import seaborn as sns
import matplotlib.pyplot as plt


def inspect_csv_data(df_date):
    """
    1.查看数据中基本信息
    :param df_date:
    :return:
    """
    print "数据集中有%d行%d列数据" % (df_date.shape[0], df_date.shape[1])
    print "*******************数据属性列信息如下************************"
    print df_date.info()
    print "*********************数据样式*******************************"
    print df_date.head(2)
    return None


def process_missing_data(df_data):
    """
    2.处理缺失数据
    :param df_data:
    :return:
    """
    if df_data.isnull().values.any():
        print "数据集中的缺失数据使用0.填充"
        df_data = df_data.fillna(0.)
    return df_data


def visualize_single_charecter(df_data, col1):
    """
     单个特征可视化
    :param df_data:
    :param col1:
    :return:
    """
    sns.boxplot(x='label', y=col1, data=df_data)
    # 作出拟合图线------hue 区分颜色的依据
    g2 = sns.FacetGrid(df_data, hue="label", size=6)
    g2.map(sns.kdeplot, col1)
    g2.add_legend()

    plt.show()


def visualize_two_charecter(df_data, col1, col2):
    """
      两个特征可视化
       :param df_data:
       :param col1:
       :param col2:
       :return:
    """
    g = sns.FacetGrid(df_data, hue='label', size=8)
    g = g.map(plt.scatter, col1, col2)
    g.add_legend()
    plt.show()
    return None


def visualize_multi_charecter(df_data, list_col):
    sns.pairplot(df_data[list_col], hue='label', size=4)
    plt.show()
