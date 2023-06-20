import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def generate_trainingdata(m=25):
    return np.array([0, 0]) + 0.25 * np.random.randn(m, 2)


def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y = 0
    count = 0
    for w in minibatch:
        z = x - w - 1
        y = y + min(33 * (z[0] ** 2 + z[1] ** 2), (z[0] + 10) ** 2 + (z[1] + 7) ** 2)
        count = count + 1
    return y / count


x0, x1, w0, w1 = sp.symbols('x0 x1 w0 w1', real=True)
# as z = x - w - 1
# y = y + min(33 * (z[0] ** 2 + z[1] ** 2), (z[0] + 10) ** 2 + (z[1] + 7) ** 2)
f = sp.Min(33 * ((x0-w0-1)**2 + (x1-w1-1)**2), (x0-w0+9)**2 + (x1-w1+6)**2)
df0 = sp.diff(f, x0)
df1 = sp.diff(f, x1)
print(df0)
print(df1)
