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