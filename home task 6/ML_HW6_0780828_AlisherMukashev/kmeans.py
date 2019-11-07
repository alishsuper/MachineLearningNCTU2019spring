# core function
import numpy as np

def k_means_plus_plus(dataset, k):
    mean = np.zeros((k, dataset.shape[1]))
    mean[0, :] = dataset[np.random.choice(range(dataset.shape[0]), 1), :]
    for j in range(1, k):
        weights = np.sum((dataset-mean[j-1])**2, axis=1)
        prob = np.cumsum(weights/np.sum(weights)) # spread probabillities from 0 to 1
        c = prob.searchsorted(np.random.rand(), 'right') # pick next center to random distance
        mean[j, :] = dataset[c, :]
    return mean

def k_means(dataset, k, mean, err):
    set_classification = []
    previous_classification = np.zeros((dataset.shape[0], k))
        
    while(True):
        weights = np.matmul(dataset**2, np.ones((dataset.shape[1], mean.shape[0]))) \
        + np.matmul(np.ones((dataset.shape[0], dataset.shape[1])), (mean**2).T) \
        - 2*np.dot(dataset, mean.T)

        classification = np.zeros(weights.shape)
        ind_min = np.argmin(weights, axis=1)
        classification[np.arange(weights.shape[0]), ind_min] = 1
        
        mean = np.matmul(classification.T, dataset) / np.sum(classification, axis=0)[:, None]

        if np.count_nonzero(np.sum(classification, axis=0) == 0):
            print('Try again!')
            raise ArithmeticError
            
        set_classification.append(classification)
            
        error = np.count_nonzero(np.abs(classification - previous_classification))
        
        if error < err:
            break
        
        previous_classification = classification
    
    return set_classification