import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from enum import Enum


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


class ALGORITHM(Enum):
    constant = 0
    polyak = 1
    rmsprop = 2
    heavyball = 3
    adam = 4


class SGD:
    def __init__(self, f, df, x, algorithm, params, batch_size, training_data):
        self.epsilon = 1e-8
        self.f = f
        self.df = df
        self.x = deepcopy(x)
        self.n = len(x)
        self.params = params
        self.batch_size = batch_size
        self.training_data = training_data
        self.records = {
            'x': [deepcopy(self.x)],
            'f': [self.f(self.x, self.training_data)],
            'step': []
        }
        self.algorithm = self.get_algorithm_via_enum_name(algorithm)
        self.initial(algorithm)

    def minibatch(self):
        # shuffle training data
        np.random.shuffle(self.training_data)
        n = len(self.training_data)
        for i in range(0, n, self.batch_size):
            if i + self.batch_size > n:
                continue
            data = self.training_data[i:(i + self.batch_size)]
            self.algorithm(data)
        self.records['x'].append(deepcopy(self.x))
        self.records['f'].append(self.f(self.x, self.training_data))

    def get_algorithm_via_enum_name(self, algorithm):
        if algorithm == ALGORITHM.constant:
            return self.constant
        elif algorithm == ALGORITHM.polyak:
            return self.polyak
        elif algorithm == ALGORITHM.rmsprop:
            return self.rmsprop
        elif algorithm == ALGORITHM.heavyball:
            return self.heavy_ball
        else:
            return self.adam

    def initial(self, algorithm):
        if algorithm == ALGORITHM.rmsprop:
            self.records['step'] = [[self.params['alpha']] * self.n]
            self.vars = {
                'sums': [0] * self.n,
                'alphas': [self.params['alpha']] * self.n
            }
        elif algorithm == ALGORITHM.heavyball:
            self.records['step'] = [0]
            self.vars = {
                'z': 0
            }
        elif algorithm == ALGORITHM.adam:
            self.records['step'] = [[0] * self.n]
            self.vars = {
                'ms': [0] * self.n,
                'vs': [0] * self.n,
                'step': [0] * self.n,
                't': 0
            }

    def constant(self, data):
        alpha = self.params['alpha']
        for i in range(self.n):
            self.x[i] -= alpha * self.get_derivative(i, data)
        self.records['step'].append(alpha)

    def polyak(self, data):
        Sum = 0
        for i in range(self.n):
            Sum = Sum + self.get_derivative(i, data) ** 2
        step = self.f(self.x, data) / (Sum + self.epsilon)
        for i in range(self.n):
            self.x[i] -= step * self.get_derivative(i, data)
        self.records['step'].append(step)

    def rmsprop(self, data):
        alpha = self.params['alpha']
        beta = self.params['beta']
        alphas = self.vars['alphas']
        sums = self.vars['sums']
        for i in range(self.n):
            self.x[i] -= alphas[i] * self.get_derivative(i, data)
            sums[i] = (beta * sums[i]) + ((1 - beta) * (self.get_derivative(i, data) ** 2))
            alphas[i] = alpha / ((sums[i] ** 0.5) + self.epsilon)
        self.records['step'].append(deepcopy(alphas))

    def heavy_ball(self, data):
        alpha = self.params['alpha']
        beta = self.params['beta']
        z = self.vars['z']
        Sum = 0
        for i in range(self.n):
            Sum += self.get_derivative(i, data) ** 2

        z = (beta * z) + (alpha * self.f(self.x, data) / (Sum + self.epsilon))
        for i in range(self.n):
            self.x[i] -= z * self.get_derivative(i, data)
        self.vars['z'] = z
        self.records['step'].append(z)

    def adam(self, data):
        alpha = self.params['alpha']
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        ms = self.vars['ms']
        vs = self.vars['vs']
        step = self.vars['step']
        t = self.vars['t']
        t += 1
        for i in range(self.n):
            ms[i] = (beta1 * ms[i]) + ((1 - beta1)*self.get_derivative(i, data))
            vs[i] = (beta2 * vs[i]) + ((1 - beta2)*(self.get_derivative(i, data) ** 2))
            _m = ms[i] / (1 - (beta1 ** t))
            _v = vs[i] / (1 - (beta2 ** t))
            step[i] = alpha * (_m / ((_v ** 0.5) + self.epsilon))
            self.x[i] -= step[i]
        self.vars['t'] = t
        self.records['step'].append(deepcopy(step))

    def get_derivative(self, i, data):
        Sum = 0
        for j in range(self.batch_size):
            Sum = Sum + self.df[i](*self.x, *data[j])
        return Sum / self.batch_size


def df0(x0, x1, w0, w1):
    return (-66 * w0 + 66 * x0 - 66) * np.heaviside(
        -33 * (-w0 + x0 - 1) ** 2 + (-w0 + x0 + 9) ** 2 - 33 * (-w1 + x1 - 1) ** 2 + (-w1 + x1 + 6) ** 2, 0) + (
                   -2 * w0 + 2 * x0 + 18) * np.heaviside(
        33 * (-w0 + x0 - 1) ** 2 - (-w0 + x0 + 9) ** 2 + 33 * (-w1 + x1 - 1) ** 2 - (-w1 + x1 + 6) ** 2, 0)


def df1(x0, x1, w0, w1):
    return (-66 * w1 + 66 * x1 - 66) * np.heaviside(
        -33 * (-w0 + x0 - 1) ** 2 + (-w0 + x0 + 9) ** 2 - 33 * (-w1 + x1 - 1) ** 2 + (-w1 + x1 + 6) ** 2, 0) + (
                   -2 * w1 + 2 * x1 + 12) * np.heaviside(
        33 * (-w0 + x0 - 1) ** 2 - (-w0 + x0 + 9) ** 2 + 33 * (-w1 + x1 - 1) ** 2 - (-w1 + x1 + 6) ** 2, 0)


colors = [
    'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:pink'
]


def helper(f, training_data, xs, legend):
    X_Data = np.linspace(-20, 20, 100)
    Y_Data = np.linspace(-20, 20, 100)
    Z_Data = []
    for x in X_Data:
        z = []
        for y in Y_Data: z.append(f([x, y], training_data))
        Z_Data.append(z)
    Z = np.array(Z_Data)
    X, Y = np.meshgrid(X_Data, Y_Data)
    plt.contour(X, Y, Z, 60)
    plt.xlabel('x0')
    plt.ylabel('x1')
    for i in range(len(xs)):
        x0 = [x[1] for x in xs[i]]
        x1 = [x[0] for x in xs[i]]
        plt.plot(x0, x1, color='dimgrey', marker='*', markeredgecolor=colors[i], markersize=3)
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
    plt.legend(legend)
    plt.show()


def c_i(f, df):
    training_data = generate_trainingdata()
    cnt = 100
    iters = list(range(cnt + 1))
    xs = []
    fs = []
    labels = ['Baseline']
    sgd_baseline = SGD(f, df, [3, 3], ALGORITHM.constant, {'alpha': 0.1}, 5, training_data)
    for _ in range(cnt):
        sgd_baseline.minibatch()
    plt.plot(iters, sgd_baseline.records['f'], label=labels[0])
    xs.append(deepcopy(sgd_baseline.records['x']))
    fs.append(deepcopy(sgd_baseline.records['f']))
    batch_sizes = [1, 5, 10, 15, 20]
    for n in batch_sizes:
        sgd = SGD(f, df, [3, 3], ALGORITHM.polyak, {}, n, training_data)
        for _ in range(cnt):
            sgd.minibatch()
        labels.append(f'batch size=${n}$')
        plt.plot(iters, sgd.records['f'], label=labels[-1])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 60])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    helper(f, training_data, xs, labels)


if __name__ == '__main__':
    c_i(f, [df0, df1])
