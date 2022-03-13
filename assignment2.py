#%%
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data1 = pd.read_csv(r"C:\Users\Alan\Desktop\資料科學\HW\assignment2\train.csv")
data2 = pd.read_csv(r"C:\Users\Alan\Desktop\資料科學\HW\assignment2\test.csv")
print(data1.shape)
# 檢查各欄位有無空值
data1.isnull().sum() 
#資料前處理
data1 = data1.drop('ID', axis=1)
data1 = data1.drop('TS', axis=1)
data2 = data2.drop('ID', axis=1)
data2 = data2.drop('TS', axis=1)

X = data1.drop('Y', axis = 1)
y = data1['Y']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print('X_train shape:',X_train.shape)
# print('X_test shape:',X_test.shape)
# print('y_train shape:',y_train.shape)
# print('y_test shape:',y_test.shape)


#正規化/標準化
from sklearn import preprocessing
def normalize(data1, method = 'MinMaxScaler'):
    '''
    method = 'StandardScaler', 'MinMaxScaler'
    '''    
    if method == 'StandardScaler':
        normalize_function = preprocessing.StandardScaler()
    if method == 'MinMaxScaler':
        normalize_function = preprocessing.MinMaxScaler()
    
    features = data1.columns
    for feature in features:
        reshape = np.array(data1[feature]).reshape(-1, 1)
        data1[feature] = normalize_function.fit_transform(reshape)

    return data1

X_train_norm = normalize(data1)
#X_test_norm = normalize(X_test)
# print(X_train_norm)

#上採樣
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
sm = SMOTE(random_state=1)
X_sm, y_sm = sm.fit_resample(X, y.astype("int"))

# upsample = RandomOverSampler(random_state=1)
# X_upsample, y_unsample = upsample.fit_resample(X, y)


print('Class 1 num: ', sum(y_sm == 1.0))
print('Class 2 num: ', sum(y_sm == 2.0))
print('Class 3 num: ', sum(y_sm == 3.0))
print('Class 4 num: ', sum(y_sm == 4.0))
print('Class 5 num: ', sum(y_sm == 5.0))
ax = sns.countplot(x="Y", data=data1)


#模型訓練
#from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# model = lr.fit(X_sm,y_sm)

# predict = model.predict(data2)
# print("predict label:",predict)

# probability = model.predict_proba(data2)
# print('predict probability:\n',probability)
# df =  pd.DataFrame(list(probability))
# # #df.to_csv('HW2-3.csv', index=False)
# print('accuracy:',lr.score(X_sm, y_sm))

from catboost import CatBoostClassifier
# cb = CatBoostClassifier(verbose=False)
# mod = cb.fit(X_sm,y_sm)

# predict = mod.predict(data2)
# print("predict label:",predict)

# probability = mod.predict_proba(data2)
# print('predict probability:\n',probability)
# df =  pd.DataFrame(list(probability))
# df.to_csv('HW2-1.csv', index=False)

# print('accuracy:',cb.score(X_sm, y_sm))



#混淆矩陣
# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
# confusion_matrix(predict, y_test)



# %%



