from timeit import timeit
import matplotlib

from a_i import random_search
from a_ii import gradient_descent
from b_i import new_global_search

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


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
    new_global_search_time1 = timeit(lambda: new_global_search(f1, n, data_range, N=New_Global_Search_N, M=New_Global_Search_M,
                                                       itr_times=New_Global_Search_n), number=500)
    global_random_search_time2 = timeit(lambda: random_search(f2, n, data_range, N=iter_times), number=500)
    gradient_descent_time2 = timeit(lambda: gradient_descent(f2, df2, n, x0, iter_times=iter_times), number=500)
    new_global_search_time2 = timeit(
        lambda: new_global_search(f2, n, data_range, N=New_Global_Search_N, M=New_Global_Search_M,
                                  itr_times=New_Global_Search_n), number=500)
    print(f'Global Random Search for f1(x) = {global_random_search_time1}')
    print(f'Gradient Descent for f1(x) = {gradient_descent_time1}')
    print(f'New Global Search for f1(x) = {new_global_search_time1}')

    print(f'Global Random Search for f2(x) = {global_random_search_time2}')
    print(f'Gradient Descent for f2(x) = {gradient_descent_time2}')
    print(f'New Global Search for f2(x) = {new_global_search_time2}')

    global_random_search_x_list, global_random_search_f_list = random_search(f1, n, data_range, N=iter_times)
    gradient_descent_x_list, gradient_descent_f_list = gradient_descent(f1, df1, n, x0, iter_times=iter_times)
    new_global_search_x_list, new_global_search_f_list = new_global_search(f1, n, data_range, New_Global_Search_N, New_Global_Search_M, New_Global_Search_n)
    global_random_search_x_list_2, global_random_search_f_list_2 = random_search(f2, n, data_range, N=iter_times)
    gradient_descent_x_list_2, gradient_descent_f_list_2 = gradient_descent(f2, df2, n, x0, iter_times=iter_times)
    new_global_search_x_list_2, new_global_search_f_list_2 = new_global_search(f1, n, data_range, New_Global_Search_N,
                                                                           New_Global_Search_M, New_Global_Search_n)

    global_random_search_x_list_ = list(range(len(global_random_search_f_list)))
    gradient_descent_x_list_ = list(range(len(gradient_descent_f_list)))
    new_global_search_x_list_ = list(range(len(new_global_search_x_list)))
    global_random_search_x_list_2_ = list(range(len(global_random_search_f_list_2)))
    gradient_descent_x_list_2_ = list(range(len(gradient_descent_f_list_2)))
    new_global_search_x_list_2_ = list(range(len(new_global_search_x_list_2)))

    plt.plot(global_random_search_x_list_, global_random_search_f_list, label='Global Random Search Algorithm',
             color='tab:blue')
    plt.plot(gradient_descent_x_list_, gradient_descent_f_list, label=f'Gradient Descent Algorithm', color='tab:red')
    plt.plot(new_global_search_x_list_, new_global_search_f_list, label=f'New Global Search Algorithm', color='tab'
                                                                                                              ':orange')
    plt.xlabel('function evaluations')
    plt.ylabel('f1')
    plt.legend()
    plt.show()

    plt.plot(global_random_search_x_list_2_, global_random_search_f_list_2, label='Search Random Algorithm',
             color='tab:blue')
    plt.plot(gradient_descent_x_list_2_, gradient_descent_f_list_2, label=f'Gradient Descent Algorithm',
             color='tab:red')
    plt.plot(new_global_search_x_list_, new_global_search_f_list, label=f'New Global Search Algorithm', color='tab'
                                                                                                              ':orange')
    plt.xlabel('function evaluations')
    plt.ylabel('f2')
    plt.legend()
    plt.show()
