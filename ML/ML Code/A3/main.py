import numpy as np
import matplotlib
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve


def ia():
    data_txt = np.loadtxt("dataset2.txt", delimiter=',')
    Degree = [1, 2, 3, 4, 5, 6, 7, 8]
    C = [0.01, 0.1, 1, 10, 100, 1000]
    # q=4
    for degree_ in Degree:
        score_list = []
        pl = PolynomialFeatures(degree=degree_)
        data_X = pl.fit_transform(data_txt[:, 0:2])
        data_Y = data_txt[:, 2]
        for ci in C:
            logistic = LogisticRegression(penalty="l2", C=ci)
            f1_score = cross_val_score(logistic, data_X, data_Y, cv=5, scoring='f1')
            score_list.append(f1_score.mean())
        max_score = max(score_list)
        print("max score of q", degree_, "=", max_score)
    C = [0.01, 0.1, 1, 10, 25, 50, 100]
    score_mean = []
    score_std = []
    data_X = PolynomialFeatures(degree=4).fit_transform(data_txt[:, 0:2])
    data_Y = data_txt[:, 2]
    for ci in C:
        logistic = LogisticRegression(penalty="l2", C=ci)
        f1_score = cross_val_score(logistic, data_X, data_Y, cv=5, scoring='f1')
        score_mean.append(f1_score.mean())
        score_std.append(f1_score.std())
    plt.axes(xscale="log")
    plt.errorbar(C, score_mean, yerr=score_std)
    plt.xlabel('C')
    plt.ylabel('F1 Score')
    plt.show()


def ib():
    # k = 20
    data_txt = np.loadtxt("dataset2.txt", delimiter=',')
    K = [1, 5, 10, 20, 30, 40, 50, 60]
    score_mean = []
    score_std = []
    data_X = data_txt[:, 0:2]
    data_Y = data_txt[:, 2]
    for k in K:
        knn = KNeighborsClassifier(n_neighbors=k)
        f1_score = cross_val_score(knn, data_X, data_Y, cv=5, scoring='f1')
        score_mean.append(f1_score.mean())
        score_std.append(f1_score.std())

    plt.axes(xscale="log")
    plt.errorbar(K, score_mean, yerr=score_std)
    plt.xlabel('K')
    plt.ylabel('F1 Score')
    plt.show()


def ic():
    data_txt = np.loadtxt("dataset1.txt", delimiter=',')
    pl = PolynomialFeatures(degree=4)
    data_X = pl.fit_transform(data_txt[:, 0:2])
    data_Y = data_txt[:, 2]

    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=40)
    logistic = LogisticRegression(penalty="l2", C=100)
    logistic.fit(X_train, Y_train)
    prediction = logistic.predict(X_test)
    cm = confusion_matrix(Y_test, prediction)
    p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '-1'])
    p.plot()
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()
    print("Logistic Regression: ")
    data_X = data_txt[:, 0:2]
    data_Y = data_txt[:, 2]
    X_train, X_test, Y_train, Y_test = train_test_split(data_txt[:, 0:2], data_txt[:, 2], test_size=0.3,
                                                        random_state=40)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, Y_train)
    prediction = knn.predict(X_test)
    cm = confusion_matrix(Y_test, prediction)
    p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '-1'])
    p.plot()
    plt.title("kNN Confusion Matrix")
    plt.show()
    print("K-Nearest Neighbours: ")
    print(confusion_matrix(Y_test, prediction))


    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, Y_train)
    prediction = baseline.predict(X_test)
    cm = confusion_matrix(Y_test, prediction)
    p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '-1'])
    p.plot()
    plt.title("Most Frequent Baseline Confusion Matrix")
    plt.show()
    print("Baseline classifier for the most frequent class: ")
    print(confusion_matrix(Y_test, prediction))

    baseline = DummyClassifier(strategy="uniform")
    baseline.fit(X_train, Y_train)
    prediction = baseline.predict(X_test)
    cm = confusion_matrix(Y_test, prediction)
    p = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '-1'])
    p.plot()
    plt.title("Uniform Baseline Confusion Matrix")
    plt.show()
    print("Baseline classifier making random predictions: ")
    print(confusion_matrix(Y_test, prediction))


def id():
    data_txt = np.loadtxt("dataset1.txt", delimiter=',')
    pl = PolynomialFeatures(degree=4)
    data_X = pl.fit_transform(data_txt[:, 0:2])
    data_Y = data_txt[:, 2]
    X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.3,
                                                        random_state=40)
    logistic = LogisticRegression(C=100).fit(X_train, Y_train)
    knn = KNeighborsClassifier(n_neighbors=20).fit(X_train, Y_train)
    fpr_1, tpr_1, _ = roc_curve(Y_test, logistic.decision_function(X_test))
    prediction_knn = knn.predict_proba(X_test)
    fpr_2, tpr_2, _ = roc_curve(Y_test, prediction_knn[:, 1])

    baseline = DummyClassifier(strategy="uniform")
    baseline.fit(X_train, Y_train)
    prediction_baseline = baseline.predict_proba(X_test)
    fpr_3, tpr_3, _ = roc_curve(Y_test, prediction_baseline[:, 1])
    plt.plot(fpr_1, tpr_1)
    plt.plot(fpr_2, tpr_2)
    plt.plot(fpr_3, tpr_3, color='green', linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(['Logistic Regression', 'KNN', 'Baseline'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    ic()
