#%%import numpy as np
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
# 读入要用到的红酒数据集
wine_data = pd.read_csv(r"C:\Users\Alan\Desktop\python code\wine.csv")
wine_data.head() 
# %%
# 查看红酒数据集的统计信息
wine_data.describe
# %%

# 定义从输入数据集中取指定列作为训练集和测试集的函数(从取1列一直到取11列):
def xattrSelect(x, idxSet):
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return(xOut)


xList = [] # 构造用于存放属性集的列表
labels = [float(label) for label in wine_data.iloc[:,-1].tolist()] # 提取出wine_data中的标签集并放入列表中
names = wine_data.columns.tolist() # 提取出wine_data中所有属性的名称并放入列表中
for i in range(len(wine_data)):
    xList.append(wine_data.iloc[i,0:-1]) # 列表xList中的每个元素对应着wine_data中除去标签列的每一行

# 将原始数据集划分成训练集(占2/3)和测试集(占1/3)：
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0 ]
xListTrain = [xList[i] for i in indices if i%3 != 0 ]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]


attributeList = [] # 构造用于存放属性索引的列表
index = range(len(xList[1])) # index用于下面代码中的外层for循环
indexSet = set(index) # 构造由names中的所有属性对应的索引构成的索引集合
oosError = [] # 构造用于存放下面代码中的内层for循环每次结束后最小的RMSE


for i in index:
    attSet = set(attributeList)
    attTrySet = indexSet - attSet # 构造由不在attributeList中的属性索引组成的集合
    attTry = [ii for ii in attTrySet] # 构造由在attTrySet中的属性索引组成的列表
    errorList = []
    attTemp = []

    for iTry in attTry:
        attTemp = [] + attributeList
        attTemp.append(iTry)

    # 调用attrSelect函数从xListTrain和xListTest中选取指定的列构成暂时的训练集与测试集
        xTrainTemp = xattrSelect(xListTrain, attTemp)
        xTestTemp = xattrSelect(xListTest, attTemp)

    # 将需要用到的训练集和测试集都转化成数组对象
        xTrain = np.array(xTrainTemp)
        yTrain = np.array(labelsTrain)
        xTest = np.array(xTestTemp)
        yTest = np.array(labelsTest)

    # 使用scikit包训练线性回归模型
        wineQModel = linear_model.LinearRegression()
        wineQModel.fit(xTrain,yTrain)

    # 计算在测试集上的RMSE
        rmsError = np.linalg.norm((yTest-wineQModel.predict(xTest)), 2)/sqrt(len(yTest)) # 利用向量的2范数计算RMSE
        errorList.append(rmsError)
        attTemp = []

    iBest = np.argmin(errorList) # 选出errorList中的最小值对应的新索引
    attributeList.append(attTry[iBest]) # 利用新索引iBest将attTry中对应的属性索引添加到attributeList中
oosError.append(errorList[iBest]) # 将errorList中的最小值添加到oosError列表中

print("Out of sample error versus attribute set size" )
print(oosError)
print("\n" + "Best attribute indices")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "Best attribute names")
print(namesList) 
# %%

# 绘制由不同数量的属性构成的线性回归模型在测试集上的RMSE与属性数量的关系图像
x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.xlabel('Number of Attributes')
plt.ylabel('Error (RMS)')
plt.show() 
# %%
