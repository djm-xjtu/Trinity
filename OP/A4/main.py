import sympy as sp
from copy import deepcopy
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def function_derivatives():
    x, y = sp.symbols('x y', real=True)
    f1 = 3 * ((x - 3) ** 4) + 9 * ((y - 8) ** 2)
    df1_x = sp.diff(f1, x)
    df1_y = sp.diff(f1, y)
    print(f1)
    print(df1_x)
    print(df1_y)
    f2 = sp.Max(x - 3, 0) + 9 * sp.Abs(y - 2)
    df2_x = sp.diff(f2, x)
    df2_y = sp.diff(f2, y)
    print(f2)
    print(df2_x)
    print(df2_y)

def polyak(x0, f, df, epsilon=1e-8):
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)], []
    for _ in range(300):
        cnt = 0
        for i in range(n):
            cnt = sum(df[i](x[i]) ** 2)
        step = f(*x) / (cnt + epsilon)
        for i in range(n):
            x[i] -= step * df[i](x[i])
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(step)
    return x_list, f_list, step_list


def RMSprop(x0, f, df, alpha0, beta):
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)], []
    epsilon = 1e-8
    sums = [0] * n
    alphas = [alpha0] * n
    for _ in range(300):
        for i in range(n):
            x[i] -= alphas[i] * df[i](x[i])
            sums[i] = (beta * sums[i]) + ((1 - beta) * (df[i](x[i]) ** 2))
            alphas[i] = alpha0 / ((sums[i] ** 0.5) + epsilon)
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(deepcopy(alphas))
    return x_list, f_list, step_list


def Heavy_Ball(x0, f, df, alpha, beta):
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)], [0]
    epsilon = 1e-8
    z = 0
    for _ in range(300):
        cnt = 0
        for i in range(n):
            cnt = sum(df[i](x[i])**2)
        z = (beta * z) + (alpha * f(*x) / (cnt + epsilon))
        for i in range(n):
            x[i] -= z * df[i](x[i])
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(z)
    return x_list, f_list, step_list


def adam(x0, f, df, alpha, beta1, beta2):
    x = deepcopy(x0)
    n = len(df)
    x_list, f_list, step_list = [deepcopy(x)], [f(*x)], [[0] * n]
    epsilon = 1e-8
    ms = [0] * n
    vs = [0] * n
    step = [0] * n
    t = 0
    for _ in range(300):
        t += 1
        for i in range(n):
            ms[i] = (beta1 * ms[i]) + ((1 - beta1) * df[i](x[i]))
            vs[i] = (beta2 * vs[i]) + ((1 - beta2) * (df[i](x[i]) ** 2))
            m_hat = ms[i] / (1 - (beta1 ** t))
            v_hat = vs[i] / (1 - (beta2 ** t))
            step[i] = alpha * (m_hat / ((v_hat ** 0.5) + epsilon))
            x[i] -= step[i]
        x_list.append(deepcopy(x))
        f_list.append(f(*x))
        step_list.append(deepcopy(step))
    return x_list, f_list, step_list
