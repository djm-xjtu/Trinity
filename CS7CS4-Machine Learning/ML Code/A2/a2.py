import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
data_txt = np.loadtxt("dataset.txt", delimiter=',')
pl = PolynomialFeatures(degree=5)

data = pl.fit_transform(data_txt[:, 0:2])[:, 1:22]

X_train, X_test, Y_train, Y_test = train_test_split(data, data_txt[:, 2], test_size=0.3, random_state=40)

C = [0.1, 1, 10, 100, 1000]
# alpha = 1 / 2*C
for i in C:
    lasso = Lasso(alpha=1/(2*i))
    lasso.fit(X_train, Y_train)
    print('C =', i)
    print('Score of the lasso regression model: ', lasso.score(X_test, Y_test))
    print('Intercept of the lasso regression model: ', lasso.intercept_)
    print('Coef of the lasso regressioon model: ', lasso.coef_)
    print('')
