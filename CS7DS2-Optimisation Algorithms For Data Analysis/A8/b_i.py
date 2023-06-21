from random import uniform
import sys


def Sort(a, b):
    return map(list, zip(*sorted(zip(a, b))))


def new_global_search(f, n, data_range, N, M, itr_times):
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
    current_x_list = []
    for i in range(N):
        tmp = []
        for j in range(n):
            tmp.append(uniform(l[j], u[j]))
        current_x_list.append(tmp)
    current_f_list = [0] * N
    for i in range(N):
        current_x = current_x_list[i]
        current_f = f(current_x[0], current_x[1])
        current_f_list[i] = current_f
        if current_f < F:
            X = current_x
            F = current_f
        x_list.append(X)
        f_list.append(F)
    opt_times = (N - M) // M
    for _ in range(itr_times):
        current_f_list, current_x_list = Sort(current_f_list, current_x_list)
        for i in range(M):
            current_x = current_x_list[i]
            for j in range(opt_times):
                x_plus = [x * uniform(0.7, 1.3) for x in current_x]
                k = M + (i * opt_times) + j
                current_x_list[k] = x_plus
                current_f_list[k] = f(x_plus[0], x_plus[1])
                if current_f_list[k] < F:
                    X = current_x_list[k]
                    F = current_f_list[k]
                x_list.append(X)
                f_list.append(F)
    return x_list, f_list
