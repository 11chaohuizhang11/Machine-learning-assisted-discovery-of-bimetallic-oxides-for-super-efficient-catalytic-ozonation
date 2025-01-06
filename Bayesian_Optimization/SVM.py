import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sklearn
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
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn.datasets import make_blobs

data = pd.read_excel("D:/133/ML/LOAD/database.xlsx")
data.head()
x = data.drop(['kobs'], axis=1)
y = data[['kobs']]
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=30)
def svc_cv(C,epsilon):
    # 交叉检验，得到的评分为贝叶斯调参优化目标
    val = cross_val_score(
        # 由于bayes优化只能优化连续超参数，因此要加上int()转为离散超参数
        SVR(C=int(C),
            epsilon = int(epsilon)),
        x_train, y_train, scoring='neg_mean_squared_error', cv=10
    ).mean()
    return val
# 规定各参数搜索范围
svc_cv = BayesianOptimization(svc_cv,
                              dict(C=(0, 1000), epsilon=(0, 2)))
svc_cv.maximize()