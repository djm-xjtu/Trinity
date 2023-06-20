import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
data_txt = np.loadtxt("dataset.txt", delimiter=',')
pl = PolynomialFeatures(degree=5)
data_X = pl.fit_transform(data_txt[:, 0:2])[:, 1:22]
data_Y = data_txt[:, 2]

C = [0.1, 1, 5, 10, 50, 100]
mean_error = []
std_error = []
for c in C:
    kf = KFold(n_splits=5)
    ridge = Ridge(alpha=1 / (2 * c))
    t = []
    for index, (train, test) in enumerate(kf.split(data_X, data_Y)):
        X = []
        Y = []
        X_test = []
        Y_test = []
        for i in train:
            X.append(data_X[i])
            Y.append(data_Y[i])
        X = np.array(X)
        Y = np.array(Y)
        ridge.fit(X, Y)
        for i in test:
            X_test.append(data_X[i])
            Y_test.append(data_Y[i])
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        prediction = ridge.predict(X_test)

        t.append(mean_squared_error(Y_test, prediction))
    mean_error.append(np.array(t).mean())
    std_error.append(np.array(t).std())

plt.errorbar(C , mean_error, yerr=std_error)
plt.xlabel('C')
plt.ylabel('Mean square error')
plt.show()