import numpy as np
from matplotlib import pyplot as plt

def RBF(u, v, g=1):
    euclidean = np.matmul(u**2, np.ones((u.shape[1], v.shape[0]))) \
    + np.matmul(np.ones((u.shape[0], u.shape[1])), (v**2).T) - 2*np.dot(u, v.T)
    return np.exp(-g * euclidean)

def get_eigen_space(dataset, k, g=1):
    U = []
    
    W = RBF(dataset, dataset, g)
    D = np.sum(W, axis=1)*np.eye(W.shape[0])
    L = D - W

    eigenvalues, eigenvectors = np.linalg.eig(L)
    sorted_idx = np.argsort(eigenvalues)
    for i in range(k):
        evc = eigenvectors[:, sorted_idx[i]]
        U.append(evc[:, None])
    U = np.concatenate(U, axis=1)
    return U

def show_eigen_space(k, dataset, classification):
    ax = plt.axes()
    colors = ['g', 'y']
    for i in range(0, k):
        for j in range(0, dataset.shape[0]):
            if classification[j] == i:
                ax.scatter(dataset[j][0], dataset[j][1], c=colors[i])
    ax.title.set_text('Eigenspace of graph Laplacian')
    plt.show()