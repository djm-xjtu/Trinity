import matplotlib.pyplot as plt
from copy import deepcopy
from random import uniform
from tensorflow import keras
from keras import regularizers, layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from bayes_opt import BayesianOptimization
import numpy as np


def get_model_loss(batch_size, alpha, beta1, beta2, epochs):
    batch_size = int(batch_size)  # ensure random values are ints
    epochs = int(epochs)  # ensure random values are ints
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    n = 50000
    x_train = x_train[1:n]
    y_train = y_train[1:n]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1, verbose=0)
    y_predicts_test = model.predict(x_test)
    loss = CategoricalCrossentropy()
    return loss(y_test, y_predicts_test).numpy()


def get_model_loss_train(batch_size, alpha, beta1, beta2, epochs):
    batch_size = int(batch_size)  # ensure random values are ints
    epochs = int(epochs)  # ensure random values are ints
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    n = 5000
    x_train = x_train[1:n]
    y_train = y_train[1:n]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1, verbose=0)
    y_predicts_train = model.predict(x_train)
    loss = CategoricalCrossentropy()
    return loss(y_test, y_predicts_train).numpy()


def global_search_random(f, n, x_rng, N):
    l = [r[0] for r in x_rng]
    u = [r[1] for r in x_rng]
    best_x = None
    best_f = 1e20
    xs, fs = [], []
    vv = 0
    for _ in range(N):
        this_x = [uniform(l[i], u[i]) for i in range(n)]
        this_f = f(*this_x)
        if this_f < best_f:
            best_x = deepcopy(this_x)
            best_f = this_f
        xs.append(deepcopy(best_x))
        fs.append(best_f)
        print(vv, best_f, best_x)
        vv += 1
    return xs, fs


def bayesian_optimisation(f, x_rng):
    pbounds = {
        'batch_size': (x_rng[0][0], x_rng[0][1]),
        'alpha': (x_rng[1][0], x_rng[1][1]),
        'beta1': (x_rng[2][0], x_rng[2][1]),
        'beta2': (x_rng[3][0], x_rng[3][1]),
        'epochs': (x_rng[4][0], x_rng[4][1])
    }
    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )
    # Expected Improvement or Upper Confidence Bound
    # xi的默认值为0.01，这意味着算法将探索那些可能比当前最优值低0.01
    # 个标准差的区域。如果你想要更加探索性的搜索，可以适当增加xi的值。反之，如果你想要更快地收敛到最优值，可以适当减小xi的值。
    # kappa的默认值为2
    # .576，这是95 % 置信区间对应的值。它用于在探索和利用之间取得平衡。如果你的搜索过程过于保守，你可以适当减小kappa的值。反之，如果你的搜索过程过于冒险，你可以适当增加kappa的值。
    # alpha的默认值为1e-5，这是一个很小的值，用于在计算方差时加以平滑。如果你的目标函数的值非常接近，你可以适当增加alpha的值。
    # n_restarts_optimizer的默认值为5，这是用于重新启动优化器的次数。如果你的目标函数是非凸的，你可以适当增加n_restarts_optimizer的值。
    # normalize_y的默认值为False，这意味着我们不会对目标函数的值进行归一化。如果你的目标函数的值非常接近，你可以适当设置normalize_y为True。

    optimizer.maximize(
        init_points=5,
        n_iter=20,
        acq= 'ucb',
        xi=0.01,
        kappa=2.576,
        alpha=1e-5,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    xs = []
    fs = []
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        xs.append([res['params'][key] for key in pbounds.keys()])
        fs.append(res['target'])
    return xs, fs


def mini_batch():
    n = 5
    x_rng = [
        [1, 128],
        [0.001, 0.001],
        [0.9, 0.9],
        [0.999, 0.999],
        [20, 20]
    ]
    print('====================== mini batch ======================')
    print('global search random')
    gsr_xs, gsr_fs = global_search_random(get_model_loss, n, x_rng, N=50)
    print('bayesian optimisation')
    bay_xs, bay_fs = bayesian_optimisation(get_model_loss, x_rng)
    return gsr_xs, gsr_fs, bay_xs, bay_fs


def mini_batch_train():
    n = 5
    x_rng = [
        [1, 128],
        [0.001, 0.001],
        [0.9, 0.9],
        [0.999, 0.999],
        [20, 20]
    ]
    print('====================== mini batch ======================')
    print('global search random')
    gsr_xs, gsr_fs = global_search_random(get_model_loss_train, n, x_rng, N=50)
    print('bayesian optimisation')
    bay_xs, bay_fs = bayesian_optimisation(get_model_loss_train, x_rng)
    return gsr_xs, gsr_fs, bay_xs, bay_fs


def adam_params():
    n = 5
    x_rng = [
        [30, 30],
        [0.001, 0.1],
        [0.25, 0.99],
        [0.9, 0.9999],
        [20, 20]
    ]
    print('====================== adam params ======================')
    print('global search random')
    gsr_xs, gsr_fs = global_search_random(get_model_loss, n, x_rng, N=50)
    print('bayesian optimisation')
    bay_xs, bay_fs = bayesian_optimisation(get_model_loss, x_rng)
    return gsr_xs, gsr_fs, bay_xs, bay_fs


def adam_params_train():
    n = 5
    x_rng = [
        [45, 45],
        [0.1, 0.0001],
        [0.25, 0.99],
        [0.9, 0.9999],
        [20, 20]
    ]
    print('====================== adam params ======================')
    print('global search random')
    gsr_xs, gsr_fs = global_search_random(get_model_loss_train, n, x_rng, N=50)
    print('bayesian optimisation')
    bay_xs, bay_fs = bayesian_optimisation(get_model_loss_train, x_rng)
    return gsr_xs, gsr_fs, bay_xs, bay_fs


def epochs():
    n = 5
    x_rng = [
        [45, 45],
        [0.001, 0.001],
        [0.9, 0.9],
        [0.999, 0.999],
        [5, 30]
    ]
    print('====================== epochs ======================')
    print('global search random')
    gsr_xs, gsr_fs = global_search_random(get_model_loss, n, x_rng, N=50)
    print('bayesian optimisation')
    bay_xs, bay_fs = bayesian_optimisation(get_model_loss, x_rng)
    return gsr_xs, gsr_fs, bay_xs, bay_fs


def epochs_train():
    n = 5
    x_rng = [
        [45, 45],
        [0.001, 0.001],
        [0.9, 0.9],
        [0.999, 0.999],
        [5, 30]
    ]
    print('====================== epochs ======================')
    print('global search random')
    gsr_xs, gsr_fs = global_search_random(get_model_loss_train, n, x_rng, N=50)
    print('bayesian optimisation')
    bay_xs, bay_fs = bayesian_optimisation(get_model_loss_train, x_rng)
    return gsr_xs, gsr_fs, bay_xs, bay_fs


def plot_results(gsr_fs, bay_fs):
    gsr_xs, bay_xs = list(range(len(gsr_fs))), list(range(bay_fs))
    plt.plot(gsr_xs, gsr_fs, label='Global Search Random')
    plt.plot(bay_xs, bay_fs, label='Bayesian Optimisation')
    plt.xlabel('optimisation iterators')
    plt.ylabel('model loss')
    plt.legend()
    plt.show()


def grid_search(f, x_rng):
    batches_list = np.arange(x_rng[0][0], x_rng[0][1] + 1)
    alpha_list = np.linspace(x_rng[1][0], x_rng[1][1], num=11)
    beta1_list = np.linspace(x_rng[2][0], x_rng[2][1], num=11)
    beta2_list = np.linspace(x_rng[3][0], x_rng[3][1], num=11)
    epochs_list = np.arange(x_rng[4][0], x_rng[4][1] + 1)

    best_loss = np.inf
    best_x = None
    losses = []
    xs = []
    for batch_size in batches_list:
        for alpha in alpha_list:
            for beta1 in beta1_list:
                for beta2 in beta2_list:
                    for epochs in epochs_list:
                        x = [batch_size, alpha, beta1, beta2, epochs]
                        loss = f(*x)
                        xs.append(x)
                        losses.append(loss)
                        if loss < best_loss:
                            best_loss = loss
                            best_x = x
    print("Best loss:", best_loss)
    print("Best x:", best_x)

    # plot search process
    N = len(losses)
    X = list(range(N))
    plt.plot(X, losses, label='grid search')
    plt.xlabel('optimisation iterations')
    plt.ylabel('model loss')
    plt.legend()
    plt.show()

    return xs, losses


# run this tonight
def compares():
    print('====================== test dataset ======================')
    gsr_xs, gsr_fs, bay_xs, bay_fs = mini_batch()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = mini_batch()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = mini_batch()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = adam_params()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = epochs()
    plot_results(gsr_fs, bay_fs)

    print('====================== train dataset ======================')
    gsr_xs, gsr_fs, bay_xs, bay_fs = mini_batch_train()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = mini_batch_train()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = mini_batch_train()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = adam_params_train()
    plot_results(gsr_fs, bay_fs)

    gsr_xs, gsr_fs, bay_xs, bay_fs = epochs_train()
    plot_results(gsr_fs, bay_fs)


def chooseParameters(f):
    pbounds = {
        'batch_size': (30, 30),
        'alpha': (0.001, 0.001),
        'beta1': (0.9, 0.9),
        'beta2': (0.999, 0.999),
        'epochs': (30, 30)
    }
    optimizer_1 = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )
    optimizer_1.maximize(
        init_points=5,
        n_iter=20,
        acq='ei',
        xi=0.01,
        kappa=2.576,
        alpha=1e-5,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    optimizer_2 = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )
    optimizer_2.maximize(
        init_points=5,
        n_iter=20,
        acq='ei',
        xi=0.1,
        kappa=2.576,
        alpha=1e-5,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    optimizer_3 = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )
    optimizer_3.maximize(
        init_points=5,
        n_iter=20,
        acq='ei',
        xi=0.5,
        kappa=2.576,
        alpha=1e-5,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    fs_1 = []
    fs_2 = []
    fs_3 = []
    for i, res in enumerate(optimizer_1.res):
        print("Iteration {}: \n\t{}".format(i, res))
        fs_1.append(res['target'])
    for i, res in enumerate(optimizer_2.res):
        print("Iteration {}: \n\t{}".format(i, res))
        fs_2.append(res['target'])
    for i, res in enumerate(optimizer_3.res):
        print("Iteration {}: \n\t{}".format(i, res))
        fs_3.append(res['target'])