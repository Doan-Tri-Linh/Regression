import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from sklearn import linear_model

from sklearn.metrics import r2_score
import matplotlib

# create data

A = np.array([[34, 108, 64, 88, 99, 51, 100]]).T
B = np.array([[5, 17, 11, 8, 14, 5, 2]]).T

# create model linear
lr = linear_model.LinearRegression()

# Fit
lr.fit(A, B)

plt.plot(A, B, 'ro')

train_line = lr.predict(A)

r_square = r2_score(B, train_line)

print(r_square)
# Draw line
x0 = np.array([[1, 120]]).T
y0 = x0 * lr.coef_ + lr.intercept_
# Test
x_test = 34
y_test = x_test * lr.coef_ + lr.intercept_

print(y_test)

plt.plot(x0, y0)
plt.show()
