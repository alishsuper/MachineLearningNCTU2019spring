import numpy as np
from matplotlib import pyplot as plt

import auxiliary_function as af

mnist_X, mnist_label = af.load_dataset()
m = np.zeros((5000, 5))
for i in range(5):
    indices = (mnist_label == (i+1))
    m[indices, i] = 1
one_class_mean = np.dot(mnist_X.T, m) / 1000
mean = np.mean(mnist_X, axis = 0)
diff_mean = np.subtract(one_class_mean, mean[:, np.newaxis])
between_class_scatter = np.dot(diff_mean * 1000, diff_mean.T)
within_class = np.subtract(mnist_X.T, np.dot(one_class_mean, m.T))
within_class_scatter = np.zeros([784, 784])
for i in range(5):
    indices = (mnist_label == (i+1))
    within_class_scatter += (np.dot(within_class[:, indices], within_class[:, indices].T)) / 1000
W = af.calculate_eigenvalue(np.dot(np.linalg.pinv(within_class_scatter), between_class_scatter))
LDA = np.dot(mnist_X, W)

plt.title('LDA')
color_k = ['g', 'b', 'r', 'y', 'c']
for i in range(5):
    indices = (mnist_label == (i+1))
    plt.plot(LDA[indices, 0], LDA[indices, 1], color_k[i]+'o', label=str(i+1))
plt.legend()
plt.show()