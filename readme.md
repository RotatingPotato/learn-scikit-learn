# Scikit-learn
> [name=Chien-Hsun, Chang & Kuo-Wei, Wu]


### 什麼是 Scikit-learn
Scikit-learn 是 Python 中最流行的機器學習庫之一，它提供了各種各樣的算法，是一種機器學習的解決方案。Scikit-learn 易於使用，性能優良，並且有良好的API、文檔和支援度(Pedregosa et al., 2011)。
![SKLearn](https://hackmd.io/_uploads/ry4w-vtyp.png)

### 機器學習
機器學習(Machine Learning)，簡單來說，就是讓機器去學習，機器要如何去學習呢?
1. 篩選出正確需要的資料
2. 將資料分類
3. 訓練資料(包含訓練及測試兩階段)
4. 將訓練完成的知料模型來預測未來資料

## 使用 Scikit-learn
#### 1.安裝sckikscikit-learn
```
pip install scikit-learn
```
#### 2.匯入Scikit-learn
```
import sklearn
```
#### 3.選擇資料
這裡以Scikit-learn內建的13套玩具資料集做示範
```
from sklearn import datasets
iris = sns.load_dataset('iris')
iris.head()
```
![](https://hackmd.io/_uploads/SkcZzuKyp.png)



## 參考文獻
> 1. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(85), 2825-2830.
