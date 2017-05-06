#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 11:24
# @Author  : ywendeng
# @Version :
import matplotlib.pyplot as plt

if __name__ == '__main__':
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(12, 10), facecolor='#FFFFFF')
    plt.subplot(2,3,1)
    plt.subplot(2,3,2)
    plt.subplot(2,3,6)
    plt.show()

