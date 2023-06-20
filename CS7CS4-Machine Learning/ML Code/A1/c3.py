import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_txt = np.loadtxt("dataset.txt", delimiter=',')
data_X = []
data_Y = []
for i in range(len(data_txt)):
    data_X.append([data_txt[i][0], data_txt[i][1], data_txt[i][0]*data_txt[i][0], data_txt[i][1]*data_txt[i][1]])
    data_Y.append(data_txt[i][2])

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=40)


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, Y_train)

acc_1 = logreg.score(X_test, Y_test)

acc_2 = baseline.score(X_test, Y_test)

print('logistic regression model accuracy of prediction: ', acc_1)
print('baseline model accuracy of prediction: ', acc_2)