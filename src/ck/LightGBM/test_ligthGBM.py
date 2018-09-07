#coding=utf-8

import lightgbm as lgb
import numpy as np
import scipy

# (1)要加载 ligsvm 文本文件或 LightGBM 二进制文件到 Dataset 中
train_data = lgb.Dataset('train.svm.bin')

# (2)加载 numpy 数组到 Dataset 中
data = np.random.rand(500,10)   # 500 个样本，每个样本包含10个标签
label = np.random.randint(2,size=500)  # 二元目标变量,  0 和 1
train_data = lgb.Dataset(data,label=label)

# (3)加载 scpiy.sparse.csr_matrix 数组到 Dataset 中
# csr = scipy.sparse.csr_matrix((dat, (row, col)))
# train_data = lgb.Dataset(csr)

# 保存 Dataset 到 LightGBM 二进制文件将会使得加载更快速:
# train_data = lgb.Dataset('train.svm.txt')
# train_data.save_binary('train.bin')

print (label)






















