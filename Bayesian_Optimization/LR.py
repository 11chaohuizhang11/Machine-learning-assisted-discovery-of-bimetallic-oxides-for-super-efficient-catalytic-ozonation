import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
import sklearn
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler  # 标准化工具
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score , KFold , cross_val_predict
from sklearn.model_selection import train_test_split
import sklearn.tree as st
import sklearn.tree as tree
import sklearn.ensemble as se
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import sklearn.metrics as metrics
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
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
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=30)
LR = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1)
LR.fit(x_train, y_train)
def lr_cv(n_jobs):
    val = cross_val_score(
        LR(n_jobs=int(n_jobs)),
           x_train, y_train, scoring='neg_mean_squared_error', cv=5
    ).mean()
    return val
lr_bo = BayesianOptimization(lr_cv,
                             {'n_jobs': (1, 10)})
lr_bo.maximize()
