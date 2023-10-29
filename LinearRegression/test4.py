# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#linspace:開始值、終值和元素個數建立表示等差數列的一維陣列
xx, yy = np.meshgrid(np.linspace(0,10,20), np.linspace(0,100,20))
zz = 2.4 * xx + 4.5 * yy + np.random.randint(0,100,(20,20))
#構建成特徵、值的形式
X, Z = np.column_stack((xx.flatten(),yy.flatten())), zz.flatten()
#線性迴歸分析
regr = linear_model.LinearRegression()
regr.fit(X, Z)
#預測的一個特徵
x_test = np.array([[15.7, 91.6]])
print(regr.predict(x_test))
#畫圖視覺化分析
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xx, yy, zz,color='red') #真實點
#擬合的平面
ax.plot_wireframe(xx, yy, regr.predict(X).reshape(20,20))
ax.plot_surface(xx, yy, regr.predict(X).reshape(20,20), alpha=0.5)
plt.savefig('test4.png')
plt.show()