## 導入Python數據處理套件
import numpy as np
import pandas as pd
## 導入繪圖套件
import matplotlib.pyplot as plt
## 導入迴歸模型套件
from sklearn.linear_model import LinearRegression
## 導入多項式套件，建構多項式迴歸模型所需的套件
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
## 導入區分訓練集與測試集套件
from sklearn.model_selection import train_test_split

data=pd.read_csv("randNum.csv")

## 創建數據集
X=np.array(range(0,100)).reshape(-1,1)
y=data[['數值']]


## 訓練多項式迴歸模型
regressor = make_pipeline(PolynomialFeatures(7), LinearRegression())
regressor.fit(X,y)

## 視覺化
# plt.scatter(X,y)
plt.plot(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.savefig('test2.png')
plt.show()