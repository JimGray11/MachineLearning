#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/22 11:29
# @Author  : ywendeng
# @Version : v1.0
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
   使用线性回归来对广告投效果来做预测------使用在TV,Radio,Newspaper上的广告投放效果来
   分析和销售量之间的关系
"""

"""
 1.数据的加载
 2.显示原数据
 3.模型训练
 4.模型预测
 5.误差分析
 6.显示预测图
"""
if __name__ == "__main__":
    path = u'dataset\\advertisement\\4.Advertising.csv'
    # 1.Python自带库
    # f = file(path, 'rb')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()
    # 2.numpy读入
    # p = np.loadtxt(path, delimiter=",", skiprows=1)
    # print p
    # 3.使用pandas 读入
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    # 绘制图1
    # plt.plot(data["TV"], y, 'ro', label='TV')
    # plt.plot(data["Radio"], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'b*', label='Newspaer')
    # plt.legend(loc='lower right')  # legend 表示图例
    # plt.grid()
    # plt.show()
    # 绘制图2
    plt.figure(figsize=(9, 12))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # 获取一个模型
    liner_regression = LinearRegression()
    # 对模型中参数进行训练
    model = liner_regression.fit(x_train, y_train)
    print liner_regression
    print liner_regression.coef_
    # 表示模型中的截距
    print liner_regression.intercept_

    # 使用模型做预测
    y_hat = liner_regression.predict(x_test)
    mse = np.average((y_hat - y_test) ** 2)  # 方差
    rmse = np.sqrt(mse)  # 标准差
    print mse, rmse
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
