import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


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


T = generate_trainingdata()
x = np.linspace(start=-20, stop=20, num=100)
y = np.linspace(start=-20, stop=20, num=100)
z = []
for i in x:
    k = []
    for j in y:
        params = [i, j]
        k.append(f(params, T))
    z.append(k)
z = np.array(z)
x, y = np.meshgrid(x, y)
contour_ax = plt.subplot(111, projection='3d')
contour_ax.contour3D(x, y, z, 60)
contour_ax.set_xlabel('X1')
contour_ax.set_ylabel('X2')
contour_ax.set_zlabel('F(x, T)')
contour_ax.set_title('Contour plot')
wireFrame_ax = plt.subplot(111, projection='3d')
wireFrame_ax.plot_wireframe(x, y, z, rstride=10, cstride=5)
wireFrame_ax.set_xlabel('X1')
wireFrame_ax.set_ylabel('X2')
wireFrame_ax.set_zlabel('F(x, T)')
wireFrame_ax.set_title('WireFrame plot')
plt.show()
