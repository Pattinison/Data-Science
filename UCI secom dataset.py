import numpy as np 
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
# plt.style.use('fivethirtyeight')

# for modeling 
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
# from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import OneClassSVM

# to avoid warnings
import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")
data = pd.read_csv(r"C:\Users\Alan\OneDrive\桌面\資料科學\uci-secom.csv")
#print(data)

#顯示資料資訊
#data.info() 


#資料分割 訓練集、測試集
from sklearn.model_selection import train_test_split
train_data , test_data =train_test_split(data,random_state=777,train_size=0.8)
#print(train_data)
#print(test_data)

#觀察有多少遺失值
missing_values = data.isnull().sum()
#print(missing_values[0:592])

total_missing = missing_values.sum()
print(total_missing)

#移除缺失值、把行都相同數值去除
def preprocess_inputs(df):      #定義函數名稱
    df = df.copy()
    
    # 刪除Time行
    df = df.drop('Time', axis=1)
    
    # Drop columns with more than 25% missing values
    missing_value_columns = df.columns[df.isna().mean() >= 0.25]
    df = df.drop(missing_value_columns, axis=1)
    
    # 填補遺失值 使用平均數
    for column in df.columns:
        df[column] = df[column].fillna(df[column].mean())
    
    # Remove columns with only one value
    single_value_columns = [
        '5', '13', '42', '49', '52', '69', '97', '141', '149', '178', '179', '186', '189', '190',
        '191', '192', '193', '194', '226', '229', '230', '231', '232', '233', '234', '235', '236',
        '237', '240', '241', '242', '243', '256', '257', '258', '259', '260', '261', '262', '263',
        '264', '265', '266', '276', '284', '313', '314', '315', '322', '325', '326', '327', '328',
        '329', '330', '364', '369', '370', '371', '372', '373', '374', '375', '378', '379', '380',
        '381', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '414',
        '422', '449', '450', '451', '458', '461', '462', '463', '464', '465', '466', '481', '498',
        '501', '502', '503', '504', '505', '506', '507', '508', '509', '512', '513', '514', '515',
        '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538'
    ]
    df = df.drop(single_value_columns, axis=1)
    
    # Give text labels to the training examples
    df['Pass/Fail'] = df['Pass/Fail'].replace({-1: "PASS", 1: "FAIL"})
    
    # Split df into X and y
    y = df['Pass/Fail']
    X = df.drop('Pass/Fail', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = preprocess_inputs(data)

#print(X_train)


y_train.value_counts()
print(y_train)

fig = y_train.pie(
    y_train.value_counts(),
    values='Pass/Fail',
    names=["PASS", "FAIL"],
    title="Class Distribution",
    width=500
)

#fig.show()

def evaluate_model(model, X_test, y_test):
    
    acc = model.score(X_test, y_test)
    print("Accuracy: {:.2f}%".format(acc * 100))
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=['PASS', 'FAIL'])
    clr = data(y_test, y_pred, labels=['PASS', 'FAIL'])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=[0.5, 1.5], labels=["PASS", "FAIL"])
    plt.yticks(ticks=[0.5, 1.5], labels=["PASS", "FAIL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    print(plt.show())

