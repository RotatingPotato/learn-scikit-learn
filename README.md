# Scikit-learn
> [name=Chien-Hsun, Chang & Kuo-Wei, Wu]
> National Taichung University of Science and Technology, Taichung, Taiwan.

### 什麼是 Scikit-learn
Scikit-learn 是 Python 中最流行的機器學習套件之一，它提供了各種各樣的算法，是一種機器學習的解決方案。Scikit-learn 易於使用，性能優良，並且有良好的API、文檔和支援度(Pedregosa et al., 2011)。
![SKLearn](https://hackmd.io/_uploads/ry4w-vtyp.png)

### Scikit-learn 和其他套件的差異
- **Scikit-learn** 是廣泛使用的開源機器學習套件，建立在 NumPy、SciPy、Matplotlib 和 pandas 等常用套件之上，使其既易於訪問又具有多功能性。Scikit-learn 適合初學者使用或開發使用非神經網絡算法的東西。以一個小型的或探索性的項目，並且不需要大量的數據，就適合使用 Scikit-learn 開發。
- **TensorFlow** 是專門用於深度學習和神經網絡的開源機器學習套件。如透過 GPU 和 TPU 進行深度學習，或者在電腦叢集上（Scikit-learn 不支持），可以考慮使用 TensorFlow。
- **PyTorch** 也是深度學習軟件套件。如果正在開發的應用程序有計算密集型任務，如自然語言處理、計算機視覺等，那麼可以考慮使用 PyTorch。
- **Keras** 是一個高級深度學習框架，它將許多低級詳細訊息和計算交給 TensorFlow 處理，從而降低了代碼複雜性。如果你的應用/模型需要使用神經網絡來從大量數據中學習，那麼可以考慮使用 Keras。

#### scikit-learn 更適合於機器學習和數據分析的原因主要有以下幾點：
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

## 使用 Scikit-learn
隨著 Scikit-learn 文件測試並練習使用 Scikit-learn 套件
+ [GitHub Repo](https://github.com/RotatingPotato/learn-scikit-learn)
#### 1.安裝sckikscikit-learn
```bash
$pip install scikit-learn
```
#### 2.匯入Scikit-learn
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
由於我們建立的資料是一維的陣列所以我們使用 `reshape(-1,1)` 將一維陣列重塑為具有單個特徵的二維陣列再進行預測資料並將資料放到`predict`變數中
```py
X=X.reshape(-1,1)
predict = model.predict(X[:50,:])
```
輸出圖片
```py
plt.plot(X,predict,c="red")
plt.scatter(X,y)
plt.show()
```
![](https://hackmd.io/_uploads/SkTxfFFk6.png)

上圖中的紅線就是透過線性回歸所找出的線

### 完整程式碼如下：
```py=
# 匯入套件
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# 選擇線性模型
from sklearn.linear_model import LinearRegression

# 產生資料
rng = np.random.RandomState(42)
x = 50 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

# 建立模型
model=LinearRegression(fit_intercept=True)

# 訓練模型
X = x[:, np.newaxis]
X.shape
model.fit(X, y)

# 預測
xfit = np.linspace(-1, 50)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# 繪圖
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
```

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
