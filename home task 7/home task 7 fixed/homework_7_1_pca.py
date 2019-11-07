import numpy as np
from matplotlib import pyplot as plt

import auxiliary_function as af

mnist_X, mnist_label = af.load_dataset()
W = af.calculate_eigenvalue(np.cov(mnist_X.T))
PCA = np.dot(mnist_X, W)
plt.title('PCA')
color_k = ['g', 'b', 'r', 'y', 'c']
for i in range(5):
    indices = (mnist_label == (i+1))
    plt.plot(PCA[indices, 0], PCA[indices, 1], color_k[i]+'o', label=str(i+1))
plt.legend()
plt.show()