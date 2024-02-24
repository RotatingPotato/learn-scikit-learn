import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

dataset=pd.read_csv('data/'+os.listdir('data')[0])
print(dataset)
df = pd.DataFrame(dataset)
imputer = SimpleImputer(strategy='mean') #以平均值填補缺失值
data = imputer.fit_transform(df)
data = pd.DataFrame(data)
print(pd.DataFrame(data))
## 組合成DataFrame格式
X = data[[0:3]]
y = data[[4]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
poly= PolynomialFeatures(degree=3)
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
scatter = ax.scatter(data[1], data[2], data[3], c=data[4], cmap=cmap, s=40, marker='o', edgecolors='none', alpha=0.8)

# 設置坐標軸標籤
ax.set_xlabel('YearsExperience')
ax.set_ylabel('EducationLevel')
ax.set_zlabel('City')

# 創建颜色條
cb = plt.colorbar(scatter,pad=0.1)
cb.set_label('Salary')

# 保存為png
plt.savefig('test.png')

# 顯示圖形
plt.show()

joblib.dump(model, 'model.pkl')