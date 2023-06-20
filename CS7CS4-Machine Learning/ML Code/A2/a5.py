import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
data_txt = np.loadtxt("dataset.txt", delimiter=',')

pl = PolynomialFeatures(degree=5)
data = pl.fit_transform(data_txt[:, 0:2])[:, 1:22]

grid = []
grid_ = np.linspace(-1.5, 1.5)
for i in grid_:
    for j in grid_:
        grid.append([i, j])

grid = np.array(grid)

C = [0.1, 1, 10, 100, 1000]
for c in C:
    ridge = Ridge(alpha=1/(2*c))
    ridge.fit(data, data_txt[:, 2])

    grid = pl.fit_transform(grid[:, 0:2])[:, 1:22]
    Y = ridge.predict(grid)[:, np.newaxis]

    ax = plt.subplot(111, projection='3d')
    ax.scatter(data_txt[:, 0], data_txt[:, 1], data_txt[:, 2], color='red')
    ax.scatter(grid[:, 0], grid[:, 1], Y, color='lightgreen')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['training data', 'prediction'], loc='upper right')
    plt.show()