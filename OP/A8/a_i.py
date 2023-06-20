import sys
from random import uniform


def random_search(f, n, data_range, N):
    l = []
    u = []
    for Range in data_range:
        l.append(Range[0])
    for Range in data_range:
        u.append(Range[1])
    X = 0
    F = sys.maxsize
    x_list = []
    f_list = []
    for i in range(N):
        data_x = []
        for j in range(n):
            x = uniform(l[j], u[j])
            data_x.append(x)
        data_f = f(data_x[0], data_x[1])
        if data_f < F:
            X = data_x
            F = data_f
        x_list.append(X)
        f_list.append(F)
    return x_list, f_list
