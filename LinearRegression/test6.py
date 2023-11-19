import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

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