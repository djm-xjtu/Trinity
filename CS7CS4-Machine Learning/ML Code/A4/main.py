import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from tensorflow import keras
import sys
def iaInputMatrix():
    n = int(input())
    a = input()
    a = a.split(' ')
    arr = []
    t = []
    for i in range(len(a)):
        if int(i / n) == int((i + 1) / n):
            t.append(a[i])
        else:
            t.append(a[i])
            arr.append(t)
            t = []
    return arr


def iaCalc(arr, kernel):
    ans = []
    n = len(arr)
    k = len(kernel)
    for i in range(n - k + 1):
        t = []
        for j in range(n - k + 1):
            tmp = 0
            for p in range(k):
                for q in range(k):
                    tmp += int(arr[i + p][j + q]) * int(kernel[p][q])
            t.append(tmp)
        ans.append(t)
    # print('result =', ans)
    return ans


def ib():
    im = Image.open('img/ib2.png')
    rgb = np.array(im.convert('RGB'))
    r = rgb[:, :, 0]
    Image.fromarray(np.uint8(r)).show()
    kernel1 = iaInputMatrix()
    kernel2 = iaInputMatrix()
    result1 = iaCalc(r, kernel1)
    result2 = iaCalc(r, kernel2)
    print(result1)

def iibi():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n=5000
    num_classes = 10
    x_train = x_train[1:n]
    y_train = y_train[1:n]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model_sk = DummyClassifier(strategy="most_frequent")
    model_sk.fit(x_train, y_train)
    preds = model_sk.predict(x_test)
    preds = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, preds))

def func():
    min_val = 100000000
    max_val = -100000000
    while True:
        c = input()
        if c == "done":
            break
        elif c.isdigit():
            min_val = min(min_val, int(c))
            max_val = max(max_val, int(c))
        else:
            print("invalid input")
    print("Maximum is", max_val)
    print("Minimum is", min_val)
if __name__ == '__main__':
    func()
