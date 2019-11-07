import numpy as np

def load_dataset():
    mnist_X = np.genfromtxt('mnist_X.csv', delimiter=',')
    mnist_label = np.genfromtxt('mnist_label.csv', delimiter=',')
    return mnist_X, mnist_label

def calculate_eigenvalue(weights, dim=2):
    eigenvalues, eigenvectors = np.linalg.eigh(weights)
    sorted_idx = np.argsort(eigenvalues[::-1])
    U = []
    for i in range(dim):
        evc = eigenvectors[:, sorted_idx[i]]
        U.append(evc[:, None])
    U = np.concatenate(U, axis=1)
    return U