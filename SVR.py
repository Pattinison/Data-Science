#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


data = pd.read_csv(r"C:\Users\Alan\Desktop\python code\WIPMOVE\LK3.csv")
data.head()
x = data['WIP']
y = data['MOVE']
print(x)
print(y.shape)

# plot
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%



#%%
#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.8)

# print(x_train.shape)
# print(y_test.shape)

#%%
from sklearn.neighbors import KNeighborsRegressor

x = x.reshape(-1,1)
# 建立KNN模型
knnModel = KNeighborsRegressor(n_neighbors=3)
# 使用訓練資料訓練模型
knnModel.fit(x,y)
# 使用訓練資料預測
predicted= knnModel.predict(x)
print('R2 score: ',knnModel.score(x,y))
mse = metrics.mean_squared_error(y, predicted)
print('MSE score: ', mse)

# plot
plt.scatter(x, y, s=10, label='True')
plt.scatter(x, predicted, color="r",s=10, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# %%
from sklearn import svm
# 建立 kernel='rbf' 模型
rbfModel=svm.SVR( C=3 , kernel='rbf', gamma='auto')
# 使用訓練資料訓練模型
rbfModel.fit(x, y)
# 使用訓練資料預測分類
predicted=rbfModel.predict(x)

# 計算訓練集 MSE 誤差
mse = metrics.mean_squared_error(y, predicted)
print('訓練集 MSE: ', mse)
print('R2 score: ',rbfModel.score(x,y))

x_test = np.linspace(-0.1,1.1,500)[:,None]
predicted=rbfModel.predict(x_test)
plt.scatter(x.ravel(),y)
plt.plot(x_test,predicted,label='kernel=RBF', color='r')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%


# %%
