import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data=datasets.load_wine().data
target=datasets.load_wine().target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.25, random_state = 0)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

regr_model = LinearRegression()
regr_model.fit(data_train, target_train)

predictions = regr_model.predict(data_test)
print(predictions.round(1))
print(target_test)

print(regr_model.score(data_train, target_train).round(3))
print(regr_model.score(data_test, target_test).round(3))

x = np.arange(predictions.size)
y = x*0

plt.scatter(x,predictions-target_test)
plt.plot(x,y,color='red')
plt.savefig('test3.png')
plt.show()

print(mean_absolute_error(target_test, predictions).round(3))

print(regr_model.coef_.round(2))
print(regr_model.intercept_.round(2))