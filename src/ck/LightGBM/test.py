#coding=utf-8

import pandas as pd





df_train_X, df_train_Y = data_process(train_n)
train_data = lgb.Dataset(df_train_X, label=df_train_Y)
param = {'num_leaves': 10, 'num_trees': 50, 'objective': 'regression'}
gbm=lgb.train(param, train_data)
y_hat=gbm.predict(df_train_X)
MAE = np.mean(abs(y_hat - df_train_Y))
MSE = np.mean((y_hat - df_train_Y) ** 2)
R2 = 1-np.sum((y_hat - df_train_Y) ** 2)/np.sum((df_train_Y-np.mean(df_train_Y))**2)


























