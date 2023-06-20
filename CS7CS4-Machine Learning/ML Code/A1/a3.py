import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
data_txt = np.loadtxt("dataset.txt", delimiter=',')
x = []
data_X = []
data_Y = []
for i in range(len(data_txt)):
    data_X.append([data_txt[i][0], data_txt[i][1]])
    data_Y.append(data_txt[i][2])
    x.append(data_txt[i][0])

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

ans = logreg.predict(X_test)
prepro = logreg.predict_proba(X_test)
acc = logreg.score(X_test, Y_test)

intercept = logreg.intercept_[0]
coef_1 = logreg.coef_[0][0]
coef_2 = logreg.coef_[0][1]
print(intercept)
y = []
for i in x:
    y.append((-coef_1 * i - intercept) / coef_2)


data_A_X = []
data_A_Y = []
data_B_X = []
data_B_Y = []
pre_A_X = []
pre_A_Y = []
pre_B_X = []
pre_B_Y = []
for i in range(len(data_txt)):
    if data_txt[i][2] == 1:
        t = data_txt[i]
        data_A_X.append(t[0])
        data_A_Y.append(t[1])
    else:
        t = data_txt[i]
        data_B_X.append(t[0])
        data_B_Y.append(t[1])
for i in range(len(ans)):
    if ans[i] == 1.0:
        pre_A_X.append(X_test[i][0])
        pre_A_Y.append(X_test[i][1])
    else:
        pre_B_X.append(X_test[i][0])
        pre_B_Y.append(X_test[i][1])

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.scatter(data_A_X, data_A_Y, alpha=0.6, c='blue')
plt.scatter(data_B_X, data_B_Y, alpha=0.6, c='green')
plt.scatter(pre_A_X, pre_A_Y, alpha=0.6, c='red', marker="+")
plt.scatter(pre_B_X, pre_B_Y, alpha=0.6, c='black', marker="_")
plt.plot(x, y)
plt.legend(['1','-1', 'predict_1', 'predict_-1'],loc='upper right')
plt.show()