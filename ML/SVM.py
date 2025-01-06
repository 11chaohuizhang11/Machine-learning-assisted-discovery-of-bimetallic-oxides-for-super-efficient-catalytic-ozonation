import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sklearn
import shap
import matplotlib.colors as colors
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
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_excel("D:/133/ML/LOAD/database.xlsx")
data.head()
x = data.drop(['kobs'], axis=1)
y = data[['kobs']]
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=2)
model = SVR(C=700, gamma=0.6, epsilon=0.1)
model_pre = model.fit(x_train, y_train)
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
plt.figure(figsize=(10, 10))
plt.scatter(y_test, pre_test, color='blue', label='test')
plt.scatter(y_train, pre_train, color='red', label='train')
plt.legend()
plt.savefig('D:/133/ML/LOAD/SVM/SVM39.tif',dpi=300)
y_test = pd.DataFrame(data=y_test)
pre_test = pd.DataFrame(data=pre_test)
pre_train = pd.DataFrame(data=pre_train)
#explainer = shap.KernelExplainer(model.predict, x)
#shap_values = explainer.shap_values(x)
#feature_lable=['ENB/A', 'RB/A', 'CNB/A', 'SG', 'EAH', 'BG', 'N', 'D']
#plt.rcParams['font.family']='serif'
#plt.rcParams['font.serif']='Times New Roman'
#plt.rcParams['font.size']=28
#cmap=colors.LinearSegmentedColormap.from_list('custom',[(0, 'orange'), (1, 'green')])
#shap.summary_plot(shap_values, x, feature_names=feature_lable, cmap=cmap)
#plt.show()
#plt.gcf().set_size_inches(7,6)
#plt.savefig('/mnt/sg001/home/fz_nankai_cyj/ZCH/ML/SVM/shap.tif',dpi=300)
y_test.to_excel('D:/133/ML/LOAD/SVM/SVM38_test.xlsx')
pre_test.to_excel('D:/133/ML/LOAD/SVM/SVM38_pred.xlsx')
pre_train.to_excel('D:/133/ML/LOAD/SVM/SVM38_pred_train.xlsx')
y_train.to_excel('D:/133/ML/LOAD/SVM/SVM38_y_train.xlsx')
#可视化
#plt.scatter(x_test[:, 0], x_test[:, 1], c=pre_test)
#lw=2
#svr_rbf=SVR(kernel="rbf",C=1, gamma=0.6, epsilon=0.2)
#svrs=svr_rbf
#d=SVR.set_decision_function_request(x_test)
#plt.figure(figsize=(12,9))
#ax=plt.subplot(111, projection="3d")
#ax.contourf(x, d)
#plt.show()

