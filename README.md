# Scikit-learn
> [name=Chien-Hsun, Chang & Kuo-Wei, Wu]
> National Taichung University of Science and Technology, Taichung, Taiwan.

**[ 目錄 ]**
> [TOC]

### 什麼是 Scikit-learn
Scikit-learn 是 Python 中最流行的機器學習套件之一，它提供了各種各樣的算法，是一種機器學習的解決方案。Scikit-learn 易於使用，性能優良，並且有良好的API、文檔和支援度(Pedregosa et al., 2011)。
![SKLearn](https://hackmd.io/_uploads/ry4w-vtyp.png)

### Scikit-learn 和其他套件的差異
- **Scikit-learn** 是廣泛使用的開源機器學習套件，建立在 NumPy、SciPy、Matplotlib 和 pandas 等常用套件之上，使其既易於訪問又具有多功能性。Scikit-learn 適合初學者使用或開發使用非神經網絡算法的東西。以一個小型的或探索性的項目，並且不需要大量的數據，就適合使用 Scikit-learn 開發。
- **TensorFlow** 是專門用於深度學習和神經網絡的開源機器學習套件。如透過 GPU 和 TPU 進行深度學習，或者在電腦叢集上（Scikit-learn 不支持），可以考慮使用 TensorFlow。
- **PyTorch** 也是深度學習軟件套件。如果正在開發的應用程序有計算密集型任務，如自然語言處理、計算機視覺等，那麼可以考慮使用 PyTorch。
- **Keras** 是一個高級深度學習框架，它將許多低級詳細訊息和計算交給 TensorFlow 處理，從而降低了代碼複雜性。如果你的應用/模型需要使用神經網絡來從大量數據中學習，那麼可以考慮使用 Keras。

#### Scikit-learn 更適合於機器學習和數據分析的原因主要有以下幾點：
- **廣泛的監督學習算法**：Scikit-learn 包含了從線性回歸到隨機梯度下降（SGD）、決策樹、隨機森林等所有你可能聽說過的監督機器學習算法。
- **無監督學習算法**：包括主成分分析（PCA）、聚類、無監督神經網絡和因子分析等範圍廣泛的機器學習算法。
- **交叉驗證**：Scikit-learn 提供了多種方法來測試監督模型在未見數據上的準確性。
- **數據預處理**：數據預處理是機器學習過程中的第一步也是最關鍵的步驟，包括特徵提取和正規化。
- **模型選擇**：Scikit-learn 提供了所有可證明的算法和方法在易於訪問的 API 中。
- **易於使用和文檔完善**：Scikit-learn 的文檔做得很好，這使得它對於新手和有經驗的數據科學家來說都是一個很好的起點。

### 機器學習
機器學習(Machine Learning)，簡單來說，就是讓機器去學習，機器要如何去學習呢?
1. 篩選出正確需要的資料
2. 將資料分類
3. 訓練資料(包含訓練及測試兩階段)
4. 將訓練完成的知料模型來預測未來資料

## 使用 Scikit-learn 做線性迴歸
隨著 Scikit-learn 文件測試並練習使用 Scikit-learn 套件
+ [GitHub Repo](https://github.com/RotatingPotato/learn-scikit-learn)

### 單變數分析

#### 1.安裝 Scikit-learn
```bash
$ pip install scikit-learn
```
#### 2.匯入 Scikit-learn
```py
import sklearn    
```
#### 3.選擇資料
這邊我們利用 [NumPy](https://numpy.org/) 建立一個簡單的隨機資料並搭配 [Matplotlib](https://matplotlib.org/) 來繪製圖形
```py 
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 50 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
plt.show() 
```
![](https://hackmd.io/_uploads/Bk1m3_FJT.png)

#### 4.選擇模型
因為我們要做線性迴歸所以這邊我們選擇線性模型
```py
from sklearn.linear_model import LinearRegression
```

#### 5.建立模型
```py
model = LinearRegression()
```
將資料放進模型內訓練：
```py
model.fit(X,y)
```
#### 6.簡易預測
由於我們建立的資料是一維的陣列所以我們使用 `reshape(-1,1)` 將一維陣列重塑為具有單個特徵的二維陣列再進行預測資料並將資料放到`predict`變數中
```py
X=X.reshape(-1,1)
predict = model.predict(X[:50,:])
```
#### 7.結果
我們可以把結果也印出來方便我們查看
```py
print("模型參數：\n", model.coef_, model.intercept_)
print("\n預測結果：\n", yfit)
```
輸出圖片
```py
plt.plot(X,predict,c="red")
plt.scatter(X,y)
plt.show()
```
![](https://hackmd.io/_uploads/SkTxfFFk6.png)

上圖中的紅線就是透過線性迴歸所找出的線

### 完整程式碼如下：
```py=
# 匯入套件
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 選擇線性模型
from sklearn.linear_model import LinearRegression

# 產生資料
rng = np.random.RandomState(42)
x = 50 * rng.rand(1000)
y = 2 * x - 1 + rng.randn(1000)

# 建立模型
model = LinearRegression(fit_intercept=True)

# 訓練模型
X = x[:, np.newaxis]
X.shape
model.fit(X, y)

# 預測
xfit = np.linspace(-1, 50)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# 印出結果
print("模型參數：\n", model.coef_, model.intercept_)
print("\n預測結果：\n", yfit)

# 繪圖
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
```

### 多項式回歸

#### 所需套件
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
```
#### 導入數據
```py
data = pd.read(xxx.csv)
X = data[['xxx'],['xxx']...['xxx']]
y= data['xxx']
```
![](https://hackmd.io/_uploads/HJgoUP8Za.png)

#### 拆分訓練集與測試集
設定 `test_size = 0.3` 訓練集與測試集的比例 0.3=訓練集:測試集=7:3
`random_state = 0` 代表隨機分割的次數
```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
```
#### 訓練集資料迴歸模型
`PolynomialFeatures(4)`代表多項式的次方數
合適的次方數可以讓預估結果更貼近數據但過高的次方數會造成過度擬合的情況發生

```py
regressor = make_pipeline(PolynomialFeatures(7), LinearRegression())
regressor.fit(X_train, y_train)
```
#### 迴歸模型的準確度
```py
score = regressor.score(X_test, y_test)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')
```

```
Score:  0.9046649875898924
Accuracy: 90.46649875898925%
```

#### 視覺化
```py
plt.plot(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.show()
```
![](https://hackmd.io/_uploads/SyrpUvU-T.png)

### 多元迴歸

#### 所需套件
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
```
#### 導入數據
這邊我們使用的是sklearn提供的玩具資料
```py
data=datasets.load_wine().data
target=datasets.load_wine().target
```
#### 拆分訓練集與測試集
```py
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.25, random_state = 0)
```
可以試著觀察訓練集與測試集有多少項資料
```py
print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)
```
```py
(133, 13)
(45, 13)
(133,)
(45,)
```
#### 訓練集資料迴歸模型
```py
regr_model = LinearRegression()
regr_model.fit(data_train, target_train)
```

#### 測試模型
```py
predictions = regr_model.predict(data_test)
print(predictions.round(1))
print(target_test)
```
```py
[ 0.1  2.1  0.9  0.5  0.9 -0.1 -0.1  2.2  0.7  1.1  1.5  2.  -0.2  0.5 1.9  0.9  0.1 -0.5  1.5 -0.1  0.5  0.4  0.6  0.7  1.2  1.1  0.8  1.2 0.8  2.  -0.1  0.2  1.2  0.2  0.   0.3  1.7  1.2  1.2  2.   0.1  0.2 0.9  1.   1.3]
[0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 1 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0 2 1 1 2 0 0 1 1 1]
```
#### 評估模型表現
決定係數(coefficient of determination)
決定係數的意思是自變數資料對目標變數(應變數)的解釋能力
底下的`0.919`和`0.804`代表在訓練集與測試集中自變數對目標變數(應變數)分別有91.9%和80.4%的解釋能力
兩者數值越近代表模型沒有被過度訓練
```py
print(regr_model.score(data_train, target_train).round(3))
print(regr_model.score(data_test, target_test).round(3))
```
```py
0.919
0.804
```
殘插圖
透過殘插圖可以更視覺化的觀察模型的預測能力
如果模型的預測能力越好那預測值就會越接近實際目標值
散布點就會更靠近Y=0的水平線
這種圖可以用來比較不同模型的效能
```py
x = np.arange(predictions.size)
y = x*0

plt.scatter(x,predictions-target_test)
plt.plot(x,y,color='red')
plt.show()
```
![](https://hackmd.io/_uploads/H1Y_erZba.png)

平均絕對誤差(mean absolute error,MAE)
意即預測值與實際值差距的絕對值的平均
因此這個值越接近 0 表示差距越小、預測能力越好
剛好scikit-learn的metrics模組提供了mean_absolute_error( )可以幫助我們計算兩組資料的MAE
## 參考文獻
> 1. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(85), 2825-2830
> 2. Scikit-learn vs. TensorFlow vs. PyTorch vs. Keras - Ritza Articles. https://ritza.co/articles/scikit-learn-vs-tensorflow-vs-pytorch-vs-keras/.  
> 3. Pytorch Vs Tensorflow Vs Keras: The Differences You Should Know - Simplilearn. https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article.  
> 4. What Is Scikit-learn and why use it for machine learning? - Data Courses. https://www.datacourses.com/what-is-scikit-learn-2021/.  
> 5. Six reasons why I recommend scikit-learn – O’Reilly. https://www.oreilly.com/content/six-reasons-why-i-recommend-scikit-learn/.  
> 6. 深度学习三大框架之争(Tensorflow、Pytorch和Keras) - 知乎. https://zhuanlan.zhihu.com/p/364670970.  
> 7. Top Machine Learning Tools Comparison: TensorFlow, Keras, Scikit-learn, and PyTorch - Zfort Group. https://www.zfort.com/blog/Top-Machine-Learning-Tools-Comparison-TensorFlow-Keras-Scikit-learn-PyTorch.  
> 8. Scikit Learn - Introduction - Online Tutorials Library. https://www.tutorialspoint.com/scikit_learn/scikit_learn_introduction.htm.
> 9. https://ithelp.ithome.com.tw/articles/10197248
> 10. https://github.com/chwang12341/Machine-Learning/blob/master/Linear_Regression/sklearn_learning/Linear_Regression.ipynb
