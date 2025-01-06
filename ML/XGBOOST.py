import xgboost as xgb
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance
import warnings
import sklearn.metrics
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler  # 标准化工具
import numpy as np
from sklearn.model_selection import cross_val_score , KFold , cross_val_predict
from sklearn.model_selection import train_test_split
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.tree as tree
import sklearn.ensemble as se
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from xgboost import plot_importance
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
data = pd.read_excel("D:/133/ML/LOAD/database.xlsx")
data.head()
x = data.drop(['kobs'], axis=1)
y = data[['kobs']]
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=13)
model = xgb.XGBRegressor(max_depth=2, learning_rate=0.2779, n_estimators=623)
model.fit(x_train, y_train)
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.savefig('D:/133/ML/LOAD/XGBOOST/importanceXGB3.tif',dpi=300)
plt.show()
pre_test = model.predict(x_test)
pre_train = model.predict(x_train)
r2_score_1 = cross_val_score(model, x_test, y_test, cv=5, scoring='r2')
r2_score_train = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
rmse_tr_lr=mean_squared_error(y_test,pre_test,squared=False)
print("MSE=", rmse_tr_lr)
r2 = r2_score(y_test, pre_test)
print("r2=", r2)
print("交叉验证的R2=", r2_score_1.mean())
print("交叉验证的训练集R2=", r2_score_train.mean())
plt.figure(figsize=(10,10))
plt.scatter(y_test,pre_test,color='blue',label='test')
plt.scatter(y_train,pre_train,color='red',label='train')
plt.legend()
plt.savefig('D:/133/ML/LOAD/XGBOOST/XGB10.tif',dpi=300)
y_test=pd.DataFrame(data=y_test)
pre_test=pd.DataFrame(data=pre_test)
pre_train=pd.DataFrame(data=pre_train)
y_test.to_excel('D:/133/ML/LOAD/XGBOOST/XGB2_test.xlsx')
pre_test.to_excel('D:/133/ML/LOAD/XGBOOST/XGB2_pred.xlsx')
pre_train.to_excel('D:/133/ML/LOAD/XGBOOST/XGB2_pred_train.xlsx')
y_train.to_excel('D:/133/ML/LOAD/XGBOOST/XGB2_y_train.xlsx')