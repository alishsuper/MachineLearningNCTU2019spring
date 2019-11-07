# global libraries
import numpy as np

def RBF(u, v, g=1):
    rbf = np.sum(u**2, axis=1)[:,None] - np.sum(v**2, axis=1)[None,:]
    return np.exp(-g * np.abs(rbf))

def Kernel(classification, gram_matrix):
    return (np.matmul(gram_matrix * np.eye(gram_matrix.shape[0]), np.ones((gram_matrix.shape[0], classification.shape[1]))) \
    - 2 * (np.matmul(gram_matrix, classification) / np.sum(classification, axis=0)) \
    + (np.matmul(np.ones((gram_matrix.shape[0], classification.shape[1])), np.matmul(np.matmul(classification.T, gram_matrix), classification)*np.eye(classification.shape[1])) / (np.sum(classification, axis=0)**2)))

def kernel_k_means(gram_matrix, k, err):
    set_classification = []
    
    previous_classification = np.zeros((gram_matrix.shape[0], k))
    previous_classification[np.arange(gram_matrix.shape[0]), np.random.randint(k, size=gram_matrix.shape[0])] = 1             

    while(1):
        weights = Kernel(previous_classification, gram_matrix)

        classification = np.zeros(weights.shape)
        ind_min = np.argmin(weights, axis=1)
        classification[np.arange(weights.shape[0]), ind_min] = 1
        
        if np.count_nonzero(np.sum(classification, axis=0) == 0):
            print('Try again!')
            raise ArithmeticError

        error = np.count_nonzero(np.abs(classification - previous_classification))
        
        if error < err:
            break
        
        previous_classification = classification
        set_classification.append(classification)
    
    return set_classification