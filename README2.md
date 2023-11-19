# Scikit-learn [2]
> [name=Chien-Hsun, Chang & Kuo-Wei, Wu]
> National Taichung University of Science and Technology, Taichung, Taiwan.

**[ 目錄 ]**
> [TOC]

### 前情提要
+ [Scikit-learn [1]](https://hackmd.io/@KageRyo/Scikit-learn-1)




## 多元非線性回歸
```python
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures as PF
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

rnd = np.random.RandomState(42) #設置隨機亂數
X = rnd.uniform(-3, 3, size=100)
y = np.sin(X) + rnd.normal(size=len(X)) / 3

#將X重組成一為陣列
X = X.reshape(-1,1)

#建立測試數據，均勻分佈在訓練集X的取值範圍內的一千個點
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

#原始特徵矩陣的擬合結果
LinearR = LinearRegression().fit(X, y)
#訓練資料的擬合
LinearR.score(X,y)
# 0.5361526059318595

#測試數據的擬合
LinearR.score(line,np.sin(line))
# 0.6800102369793312

#多項式擬合，設定高次項
d=5

#進行高次項轉換
poly = PF(degree=d)
X_ = poly.fit_transform(X)
line_ = poly.transform(line)

#訓練資料的擬合
LinearR_ = LinearRegression().fit(X_, y)
LinearR_.score(X_,y)
# 0.8561679370344799

#測試數據的擬合
LinearR_.score(line_,np.sin(line))
# 0.9868904451787956

d=5
#和上面展示一致的建模流程
LinearR = LinearRegression().fit(X, y)
X_ = PF(degree=d).fit_transform(X)
LinearR_ = LinearRegression().fit(X_, y)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
line_ = PF(degree=d).fit_transform(line)

#放置畫布
fig, ax1 = plt.subplots(1)

#將測試資料帶入predict接口，獲得模型的擬合效果並進行繪製
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green'
         ,label="linear regression")
ax1.plot(line, LinearR_.predict(line_), linewidth=2, color='red'
         ,label="Polynomial regression")

#將原始資料上的擬合繪製在影像上
ax1.plot(X[:, 0], y, 'o', c='k')

#其他圖形選項
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Linear Regression ordinary vs poly")
plt.tight_layout()
plt.savefig('test5.png')
plt.show()
```
![](https://hackmd.io/_uploads/BkjHRtqMT.png)




```python
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

#導入數據
data = pd.read_csv('LinearRegression/Salary_Data2.csv')

#設定參數
X = data[['YearsExperience','EducationLevel','City']]
y = data[['Salary']]

#分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 創建一個PolynomialFeatures函式，指定多項式的次數
poly = PolynomialFeatures(3)
# 將線性特徵轉換為多項式特徵
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 創建一個多元多項式回歸模型
model = LinearRegression()

# 訓練模型
model.fit(X_train_poly, y_train)

# 預測
y_pred = model.predict(X_test_poly)

# 計算模型的性能
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5

# 查看預測結果
print(y_pred)
print("R方：", r2)
print("RMSE：", rmse)
```
```python
[[68.3       ]
 [41.21890485]
 [67.86942388]
 [82.17480962]
 [68.3       ]
 [49.52571866]
 [30.99001209]
 [26.48199526]
 [39.37658661]]
R方： 0.8919842644333875
RMSE： 5.4232823419435565
```
以下是不同次數所預測出的準確度
|         degree=1          |         degree=2          |         degree=3          |
|:-------------------------:|:-------------------------:|:-------------------------:|
|  R方：0.8466753935122813   |  R方：0.9104206298150077   |  R方：0.8919842644333875  |
| RMSE：6.461370291237774  | RMSE：4.938814374314465 | RMSE：5.4232823419435565 |
|       **degree=4**        |       **degree=5**        |       **degree=6**        |
| R方：0.9056010255605839  |  R方：0.911273846828835  | R方：-22.07849145322603  |
| RMSE： 5.0699343902801655 | RMSE：4.915237726551235  | RMSE：79.27247356110891  |

### 多維圖
#### 四維(X軸、Y軸、Z軸、顏色)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv('LinearRegression/Salary_Data2.csv')

X = data[['YearsExperience','EducationLevel','City']]
y = data[['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 創建一個PolynomialFeatures函式，指定多項式的次數
poly = PolynomialFeatures(4)
# 將線性特徵轉換為多項式特徵
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 創建一個多元多項式回歸模型
model = LinearRegression()

# 訓練模型
model.fit(X_train_poly, y_train)

# 預測
y_pred = model.predict(X_test_poly)

# 計算模型的性能
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5

# 查看預測結果
print(y_pred)
print("R方：", r2)
print("RMSE：", rmse)

fig = plt.figure(figsize=(10, 6))
t = fig.suptitle('Polynomial Regression', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

#使用 Salary 列作為颜色映射的依据
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green','yellow','orange', 'red'])
scatter = ax.scatter(data['YearsExperience'], data['EducationLevel'], data['City'], c=data['Salary'], cmap=cmap, s=data['Age']*2, marker='o', edgecolors='none', alpha=0.6)

# 設置坐標軸標籤
ax.set_xlabel('YearsExperience')
ax.set_ylabel('EducationLevel')
ax.set_zlabel('City')

# 創建颜色條
cb = plt.colorbar(scatter,pad=0.1)
cb.set_label('Salary')

# 保存為png
plt.savefig('LinearRegression/test7.png')

# 顯示圖形
plt.show()
```
![test7](https://hackmd.io/_uploads/B1ewwJwNa.png)
#### 五維(X軸、Y軸、Z軸、顏色、大小)
```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
data = pd.read_csv('LinearRegression/random.csv')

fig = plt.figure(figsize=(10, 6))
t = fig.suptitle('Polynomial Regression', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green','yellow','orange', 'red'])
#使用 Salary 列作為颜色映射的依据
scatter = ax.scatter(data['random2'], data['random3'], data['random4'], c=data['random5'], cmap=cmap, s=data['random1']*5, marker='o', edgecolors='none', alpha=0.8)

# 設置坐標軸標籤
ax.set_xlabel('random2')
ax.set_ylabel('random3')
ax.set_zlabel('random4')

# 創建颜色條
cb = plt.colorbar(scatter,pad=0.15)
cb.set_label('random5')

def update(frame):
    # 模擬動態效果，可根据需求更改

    ax.view_init(elev=10, azim=frame)


# 創建動畫
animation = FuncAnimation(fig, update, frames=range(0, 360 , 1), interval=50, repeat=True)
# 保存為png
plt.savefig('LinearRegression/test7_1.png')
# 保存為GIF
animation.save('LinearRegression/test7_1.gif', writer='imagemagick')
print('Done')
```
![test7_1](https://hackmd.io/_uploads/H1Utv1wNT.png)
Gif圖
https://1drv.ms/i/s!ArBGVTMnp3LagasZJyuGY08vtyj4BA?e=MVbeWu


```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv('LinearRegression/random.csv')

fig = plt.figure(figsize=(10, 6))
t = fig.suptitle('Polynomial Regression', fontsize=14)

# 散點圖
ax1 = fig.add_subplot(121, projection='3d')
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green','yellow','orange', 'red'])
scatter = ax1.scatter(data['random2'], data['random3'], data['random4'], c=data['random5'], cmap=cmap, s=data['random1'], marker='o', edgecolors='none', alpha=0.8)
ax1.set_xlabel('random2')
ax1.set_ylabel('random3')
ax1.set_zlabel('random4')

# 添加颜色條
cb1 = plt.colorbar(scatter, ax=ax1, pad=0.2)
cb1.set_label('Salary')

ax2 = fig.add_subplot(122)
ax2.scatter(data['random2'], data['random3'], c=data['random5'], s=data['random5'], marker='o', edgecolors='none', alpha=0.6, cmap=cmap)
ax2.set_xlabel('random2')
ax2.set_ylabel('random3')

cb2 = plt.colorbar(scatter, ax=ax2)
cb2.set_label('random5')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('LinearRegression/test7_2.png')
plt.show()

```
![test7_2](https://hackmd.io/_uploads/H1vP_yDE6.png)



















