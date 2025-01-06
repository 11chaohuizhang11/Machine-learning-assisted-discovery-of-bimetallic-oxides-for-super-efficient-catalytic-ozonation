import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMRegressor
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
from sklearn.model_selection import GridSearchCV
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
from sklearn.datasets import make_blobs
data = pd.read_excel("/mnt/sg001/home/fz_nankai_cyj/ZCH/databasev4.xlsx")
data.head()
x = data.drop(['ΔG[eV]'],axis=1)
y = data[['ΔG[eV]']]
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=42)
model = LGBMRegressor(objective='regression',num_leaves=5, learning_rate=0.1, n_estimators=1000,  max_depth=5)
estimator = LGBMRegressor(num_leaves=8)
#param_grid = {
    #'learning_rate': [0.01, 0.1, 1],
    #'n_estimators': [20, 40]
#}
#gbm = GridSearchCV(estimator, param_grid)
#gbm.fit(x_train, y_train)
#print('Best parameters found by grid search are:', gbm.best_params_)
model.fit(x_train,y_train)
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.savefig('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/LGBM/importanceLGBM4.tif',dpi=300)
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
plt.savefig('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/LGBM/LGBM6.tif',dpi=300)
y_test=pd.DataFrame(data=y_test)
pre_test=pd.DataFrame(data=pre_test)
pre_train=pd.DataFrame(data=pre_train)
y_test.to_excel('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/LGBM/LGBM3_test.xlsx')
pre_test.to_excel('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/LGBM/LGBM3_pred.xlsx')
y_train.to_excel('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/LGBM/LGBM3_train.xlsx')
pre_train.to_excel('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/LGBM/LGBM3_pretrain.xlsx')