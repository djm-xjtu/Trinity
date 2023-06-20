import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

data_txt = np.loadtxt("dataset.txt", delimiter=',')

ax = plt.subplot(111, projection='3d')
ax.scatter(data_txt[:, 0], data_txt[:, 1], data_txt[:, 2], color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
