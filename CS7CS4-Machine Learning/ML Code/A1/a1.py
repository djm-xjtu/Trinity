import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

data_txt = np.loadtxt("dataset.txt", delimiter=',')
data_A_X = []
data_A_Y = []
data_B_X = []
data_B_Y = []
for i in range(len(data_txt)):
    if data_txt[i][2] == 1:
        t = data_txt[i]
        data_A_X.append(t[0])
        data_A_Y.append(t[1])
    else:
        t = data_txt[i]
        data_B_X.append(t[0])
        data_B_Y.append(t[1])
print(data_A_X)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.scatter(data_A_X, data_A_Y, alpha=0.6, c='blue')
plt.scatter(data_B_X, data_B_Y, alpha=0.6, c='green')
plt.legend(['1','-1'],loc='upper right')
plt.show()
