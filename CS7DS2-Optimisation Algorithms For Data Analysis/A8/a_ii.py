from random import uniform
from timeit import timeit
import matplotlib
from a_i import random_search

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys


def gradient_descent(f, df, n, x0, iter_times, alpha=0.01):
    x = x0
    f = f(x0[0], x0[1])
    x_list = []
    f_list = []
    x_list.append(x0)
    f_list.append(f)
    for i in range(iter_times):
        for j in range(n):
            x[j] -= alpha * df[j](x[j])
        f = f(x[0], x[1])
        x_list.append(x)
        f_list.append(f)
    return x_list, f_list


def func_1():
    f = lambda x, y: 3 * (x - 3) ** 4 + 9 * (y - 8) ** 2
    dx = lambda x: 12 * (x - 3) ** 3
    dy = lambda y: 18 * y - 144
    return f, (dx, dy)


def func_2():
    f = lambda x, y: 9 * abs(y - 8) + max(0, x - 3)
    dx = lambda x: np.heaviside(x - 3, 0)
    dy = lambda y: 9 * np.sign(y - 8)
    return f, (dx, dy)


if __name__ == '__main__':
    f1, df1 = func_1()
    f2, df2 = func_2()
    n = 2
    data_range = [[4, 8], [0, 3]]
    x0 = [7, 7]
    iter_times = 200
    New_Global_Search_N, New_Global_Search_M, New_Global_Search_n = 20, 5, 10
    global_random_search_time1 = timeit(lambda: random_search(f1, n, data_range, N=iter_times), number=500)
    gradient_descent_time1 = timeit(lambda: gradient_descent(f1, df1, n, x0, iter_times=iter_times), number=500)
    global_random_search_time2 = timeit(lambda: random_search(f2, n, data_range, N=iter_times), number=500)
    gradient_descent_time2 = timeit(lambda: gradient_descent(f2, df2, n, x0, iter_times=iter_times), number=500)
    print(f'Global Random Search for f1(x) = {global_random_search_time1}')
    print(f'Gradient Descent for f1(x) = {gradient_descent_time1}')
    print(f'Global Random Search for f2(x) = {global_random_search_time2}')
    print(f'Gradient Descent for f2(x) = {gradient_descent_time2}')
    
    global_random_search_x_list, global_random_search_f_list = random_search(f1, n, data_range, N=iter_times)
    gradient_descent_x_list, gradient_descent_f_list = gradient_descent(f1, df1, n, x0, iter_times=iter_times)
    global_random_search_x_list_2, global_random_search_f_list_2 = random_search(f2, n, data_range, N=iter_times)
    gradient_descent_x_list_2, gradient_descent_f_list_2 = gradient_descent(f2, df2, n, x0, iter_times=iter_times)
    global_random_search_x_list_ = list(range(len(global_random_search_f_list)))
    gradient_descent_x_list_ = list(range(len(gradient_descent_f_list)))
    global_random_search_x_list_2_ = list(range(len(global_random_search_f_list_2)))
    gradient_descent_x_list_2_ = list(range(len(gradient_descent_f_list_2)))
    plt.plot(global_random_search_x_list_, global_random_search_f_list, label='Global Random Search Algorithm', color='tab:blue')
    plt.plot(gradient_descent_x_list_, gradient_descent_f_list, label=f'Gradient Descent Algorithm', color='tab:red')
    plt.xlabel('function evaluations')
    plt.ylabel('f1')
    plt.legend()
    plt.show()

    plt.plot(global_random_search_x_list_2_, global_random_search_f_list_2, label='Global Random Search Algorithm',
             color='tab:blue')
    plt.plot(gradient_descent_x_list_2_, gradient_descent_f_list_2, label=f'Gradient Descent Algorithm', color='tab:red')
    plt.xlabel('function evaluations')
    plt.ylabel('f2')
    plt.legend()
    plt.show()