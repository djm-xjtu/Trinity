import sympy
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def A1():
    x = sympy.symbols('x', real=True)
    f = x ** 4
    df_dx = sympy.diff(f, x)
    print(df_dx)
    return df_dx


def A2():
    df_dx = A1()
    x = sympy.symbols('x', real=True)
    func_1 = sympy.lambdify(x, df_dx)
    f = x ** 4
    delta = 0.01
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    func_2 = sympy.lambdify(x, finite_difference)
    X = np.arange(-200, 200, 10)
    Y1 = []
    Y2 = []
    for x in X:
        Y1.append(func_1(x))
        Y2.append(func_2(x))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(X, Y1, alpha=0.6, c='red')
    plt.scatter(X, Y2, alpha=0.6, c='black')
    plt.legend(['derivative value', 'finite difference'], loc='upper right')
    plt.show()


def A3():
    df_dx = A1()
    x = sympy.symbols('x', real=True)
    func_1 = sympy.lambdify(x, df_dx)
    f = x ** 4
    X = np.arange(-1, 1, 0.1)

    delta = 0.001
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    func_2 = sympy.lambdify(x, finite_difference)

    delta = 0.01
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    func_3 = sympy.lambdify(x, finite_difference)

    delta = 0.1
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    func_4 = sympy.lambdify(x, finite_difference)

    delta = 0.5
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    func_5 = sympy.lambdify(x, finite_difference)

    delta = 1
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    func_6 = sympy.lambdify(x, finite_difference)

    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    Y5 = []
    Y6 = []
    for x in X:
        Y1.append(func_1(x))
        Y2.append(func_2(x))
        Y3.append(func_3(x))
        Y4.append(func_4(x))
        Y5.append(func_5(x))
        Y6.append(func_6(x))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(X, Y1, c='black')
    plt.scatter(X, Y2, c='red')
    plt.scatter(X, Y3, c='blue')
    plt.scatter(X, Y4, c='green')
    plt.scatter(X, Y5, c='yellow')
    plt.scatter(X, Y6, c='orange')
    plt.legend(['derivative value', 'δ=0.001', 'δ=0.01', 'δ=0.1', 'δ=0.5', 'δ=1'], loc='upper right')
    plt.show()


def func_1(x):
    return x ** 4


def func_2(x):
    return 4*x**3


def func_3(x, gamma):
    return gamma*x**2


def func_4(x, gamma):
    return gamma*2*x


def func_5(x, gamma):
    return gamma*abs(x)


def func_6(x, gamma):
    if x < 0:
        return -gamma
    return gamma


def B1(func_1, func_2, x0, alpha=0.15, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array(func_1(x))
    for k in range(num_iters):
        step = alpha * func_2(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, func_1(x))
    return (X, F)


def B2(x0=1, alpha=0.1):
    (X, F) = B1(func_1, func_2, x0=x0, alpha=alpha)
    xx = np.arange(-1, 1.1, 0.1)
    # plt.plot(F)
    # plt.xlabel('iteration')
    # plt.ylabel('f(x)')
    # plt.show()
    # plt.plot(X)
    # plt.xlabel('iteration')
    # plt.ylabel('x')
    # plt.show()
    plt.step(X, func_1(X))
    plt.plot(xx, func_1(xx))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def B3():
    for x in np.arange(0.5, 1, 0.1):
        B2(x, alpha=0.1)
    # for alpha in [0.01, 0.1, 0.2, 0.5]:
    #     B2(1, alpha=alpha)


def C1():
    gamma = 0.1
    x = 1
    X_1 = np.array([x])
    F_1 = np.array(func_3(x, gamma))
    for k in range(50):
        step = 0.1 * func_4(x, gamma)
        x = x - step
        X_1 = np.append(X_1, [x], axis=0)
        F_1 = np.append(F_1, func_3(x, gamma))

    gamma = 0.5
    x = 1
    X_2 = np.array([x])
    F_2 = np.array(func_3(x, gamma))
    for k in range(50):
        step = 0.1 * func_4(x, gamma)
        x = x - step
        X_2 = np.append(X_1, [x], axis=0)
        F_2 = np.append(F_1, func_3(x, gamma))

    gamma = 1
    x = 1
    X_3 = np.array([x])
    F_3 = np.array(func_3(x, gamma))
    for k in range(50):
        step = 0.1 * func_4(x, gamma)
        x = x - step
        X_3 = np.append(X_1, [x], axis=0)
        F_3 = np.append(F_1, func_3(x, gamma))

    gamma = 2
    x = 1
    X_4 = np.array([x])
    F_4 = np.array(func_3(x, gamma))
    for k in range(50):
        step = 0.1 * func_4(x, gamma)
        x = x - step
        X_4 = np.append(X_1, [x], axis=0)
        F_4 = np.append(F_1, func_3(x, gamma))

    xx = np.arange(-1, 1.1, 0.1)

    plt.plot(F_1)
    plt.plot(F_2)
    plt.plot(F_3)
    plt.plot(F_4)
    plt.xlabel('iteration')
    plt.ylabel('f(x)')
    plt.show()

    plt.plot(X_1)
    plt.plot(X_2)
    plt.plot(X_3)
    plt.plot(X_4)
    plt.xlabel('iteration')
    plt.ylabel('x')
    plt.show()

    plt.step(X_1, func_3(X_1, 0.1))
    plt.plot(xx, func_3(xx, 0.1))
    plt.step(X_2, func_3(X_2, 0.5))
    plt.plot(xx, func_3(xx, 0.5))
    plt.step(X_3, func_3(X_3, 1))
    plt.plot(xx, func_3(xx, 1))
    plt.step(X_4, func_3(X_4, 2))
    plt.plot(xx, func_3(xx, 2))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['γ=0.1', 'γ=0.5', 'γ=1', 'γ=2'], loc='upper right')
    plt.show()


def C2():
    gamma = 0.1
    x = 1
    X_1 = np.array([x])
    F_1 = np.array(func_5(x, gamma))
    for k in range(50):
        step = 0.1 * func_6(x, gamma)
        x = x - step
        X_1 = np.append(X_1, [x], axis=0)
        F_1 = np.append(F_1, func_5(x, gamma))

    gamma = 0.5
    x = 1
    X_2 = np.array([x])
    F_2 = np.array(func_5(x, gamma))
    for k in range(50):
        step = 0.1 * func_6(x, gamma)
        x = x - step
        X_2 = np.append(X_1, [x], axis=0)
        F_2 = np.append(F_1, func_5(x, gamma))

    gamma = 1
    x = 1
    X_3 = np.array([x])
    F_3 = np.array(func_5(x, gamma))
    for k in range(50):
        step = 0.1 * func_6(x, gamma)
        x = x - step
        X_3 = np.append(X_1, [x], axis=0)
        F_3 = np.append(F_1, func_5(x, gamma))

    gamma = 2
    x = 1
    X_4 = np.array([x])
    F_4 = np.array(func_5(x, gamma))
    for k in range(50):
        step = 0.1 * func_6(x, gamma)
        x = x - step
        X_4 = np.append(X_1, [x], axis=0)
        F_4 = np.append(F_1, func_5(x, gamma))

    xx = np.arange(-1, 1.1, 0.1)
    plt.plot(F_1)
    plt.plot(F_2)
    plt.plot(F_3)
    plt.plot(F_4)
    plt.xlabel('iteration')
    plt.ylabel('f(x)')
    plt.show()

    plt.plot(X_1)
    plt.plot(X_2)
    plt.plot(X_3)
    plt.plot(X_4)
    plt.xlabel('iteration')
    plt.ylabel('x')
    plt.show()

    plt.step(X_1, func_5(X_1, 0.1))
    plt.plot(xx, func_5(xx, 0.1))
    plt.step(X_2, func_5(X_2, 0.5))
    plt.plot(xx, func_5(xx, 0.5))
    plt.step(X_3, func_5(X_3, 1))
    plt.plot(xx, func_5(xx, 1))
    plt.step(X_4, func_5(X_4, 2))
    plt.plot(xx, func_5(xx, 2))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['γ=0.1', 'γ=0.5', 'γ=1', 'γ=2'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    B2()
