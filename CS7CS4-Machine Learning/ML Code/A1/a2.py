import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data_txt = np.loadtxt("dataset.txt", delimiter=',')
data_X = []
data_Y = []
for i in range(len(data_txt)):
    data_X.append([data_txt[i][0], data_txt[i][1]])
    data_Y.append(data_txt[i][2])

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=40)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

acc = logreg.score(X_test, Y_test)
slope = logreg.coef_
intercept = logreg.intercept_

print('Accucary:', acc)
print('Slope: ', slope)
print('intercept: ', intercept)