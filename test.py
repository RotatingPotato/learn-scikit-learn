import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rng = np.random.RandomState(42)
x = 50 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)


from sklearn.linear_model import LinearRegression
model=LinearRegression(fit_intercept=True)
X = x[:, np.newaxis]
X.shape
model.fit(X, y)

xfit = np.linspace(-1, 50)

Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)

plt.show()