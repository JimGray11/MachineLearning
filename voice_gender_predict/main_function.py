#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/1/18 21:34
# @Author  : ywendeng
# @Version : v1.0
import pandas as pd
from pandas_tools import inspect_csv_data, process_missing_data, visualize_two_charecter, \
    visualize_multi_charecter,visualize_single_charecter

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
    visualize_single_charecter(df_data,'meanfun')
    # 可视化特征值-----查看两个特征之间的关系
    visualize_two_charecter(df_data, 'meanfun', 'centroid')
    # 可视化多个特征之间的关系
    fea_names = ['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']
    visualize_multi_charecter(df_data, fea_names)
    return None


if __name__ == "__main__":
    run_main()
