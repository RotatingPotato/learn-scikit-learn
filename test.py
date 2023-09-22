# 匯入套件
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 選擇線性模型
from sklearn.linear_model import LinearRegression

# 產生資料
rng = np.random.RandomState(42)
x = 50 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

# 訓練模型
model=LinearRegression(fit_intercept=True)
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