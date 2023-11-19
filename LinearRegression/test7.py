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
scatter = ax.scatter(data['YearsExperience'], data['EducationLevel'], data['City'], c=data['Salary'], cmap=cmap, s=40, marker='o', edgecolors='none', alpha=0.8)

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