import lightgbm as lgb
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sklearn
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler  # 标准化工具
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score , KFold , cross_val_predict
from sklearn.model_selection import train_test_split
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.tree as tree
from sklearn.ensemble import GradientBoostingRegressor
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
from sklearn.datasets import make_blobs
data = pd.read_excel("D:/133/ML/LOAD/database.xlsx")
data.head()
x = data.drop(['kobs'], axis=1)
y = data[['kobs']]
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaler,y,test_size=0.2,random_state=30)
def rf_cv(num_leaves, learning_rate, n_estimators, max_bin, max_depth):
    val = cross_val_score(
        LGBMRegressor(num_leaves=int(num_leaves),
                      learning_rate=int(learning_rate),
                      n_estimators=int(n_estimators),
                      max_bin=int(max_bin),
                      max_depth=int(max_depth),
                      random_state=30),
        x_train, y_train, scoring='neg_mean_squared_error', cv=10
    ).mean()
    return val
rf_bo = BayesianOptimization(rf_cv,
                             {'n_estimators': (50,1000),
                              'num_leaves': (5, 20),
                              'max_depth': (2, 10),
                              'learning_rate': (0.00001, 0.1),
                              'max_bin': (10, 1000)})
rf_bo.maximize()