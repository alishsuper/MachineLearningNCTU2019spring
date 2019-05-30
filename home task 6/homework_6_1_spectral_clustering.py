import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import random
import math

# local libraries
import common_function as cf

# iter function
def runKmeans(launcher):
    iter_count = 0
    all_c = []
    all_u = []
    for new_c, new_u in launcher:
        #IDisplay.clear_output(wait=True)
        iter_count += 1
        #cf.showClustering(data_source, new_c, new_u, title='iter [{}]'.format(iter_count))
        all_c.append(new_c)
        all_u.append(new_u)

    print('use {} counts to converge'.format(iter_count))
    
    return all_c, all_u
'''
def reshowKmeans(datas, all_c, all_u):
    for idx, (c, u) in enumerate(zip(all_c, all_u)):
        #IDisplay.clear_output(wait=True)
        cf.showClustering(datas, c, u, title='iter [{}]'.format(idx))
'''
# datapoints, mean
def Euclidean(x, y):
    """
    calculate Euclidean distance
    parameters:
        x: n1 * d, y: n2 x d
    return:
        d: n1 * n2
        
    d(i, j) = |x(i) - y(j)|^2
    """

    if len(x.shape)==1:
        x = x[None, :]
    if len(y.shape)==1:
        y = y[None, :]
    # to make the necessary shape
    return np.matmul(x**2, np.ones((x.shape[1],y.shape[0]))) \
    + np.matmul(np.ones((x.shape[0], x.shape[1])), (y**2).T) \
    - 2*np.dot(x, y.T)

def RBFkernel(gamma=1):
    """
    generate callable function for RBF kernel
    parameters:
        gamma : default is 1
    return:
        lambda(u, v)
        
    rbf(u, v) = exp(-gamma|u-v|^2)
    """
    return lambda u,v:np.exp(-1*gamma*Euclidean(u,v));

def KernelTrick(gram_m, c):
    """
    calculate Euclidean distance in feature space by kernel trick
    parameters:
        gram_m : gram matrix K(x, x)
        c : cluster vector c(i,k) = 1 if x(i) belong k clustering
    return:
        w : n*k
    """
    return (
        np.matmul(
        gram_m * np.eye(gram_m.shape[0]), 
        np.ones((gram_m.shape[0], c.shape[1])) 
    ) \
    - 2*( np.matmul(gram_m, c) / np.sum(c, axis=0) ) \
    + (np.matmul(
        np.ones((gram_m.shape[0], c.shape[1])), 
        np.matmul(np.matmul(c.T, gram_m), c)*np.eye(c.shape[1])
    ) / (np.sum(c,axis=0)**2) )
    )

def RandomCluster(n,k):
    """
    get random cluster c
    """
    c = np.zeros((n,k))
    c[np.arange(n),np.random.randint(k,size=n)] = 1             
    return c

def RandomMean(k, dim):
    """
    get random mean
    """
    return -1 + 2*np.random.random((k, dim))

def GetMeanFromCluster(datas, c):
    """
    get mean from cluster c
    """
    if np.count_nonzero(np.sum(c, axis=0) == 0):
        raise ArithmeticError
    return np.matmul(c.T, datas) / np.sum(c, axis=0)[:,None]
'''
def GetCluster(w):
    """
    get cluster from w (distance between x and u)
    """
    new_c = np.zeros(w.shape)
    new_c[np.arange(w.shape[0]),np.argmin(w, axis=1)] = 1
    return new_c
'''
def kmeans(datas, k, initial_u=None, initial_c=None, isKernel=False, converge_count = 1):
    """
    use generator to iter steps
    parameters:
        datas : data points
        k : how many cluster
        initial_u : assign mean u
        initial_c : assign cluster c
        isKernel : use kernel k-means (default is False)
    return:
        python generator
        
    next() will get cluster c and mean u
    e.g.
        c, u = next(g)
    if converge, next() will raise Error
    """
    
    # initialization
    if isKernel: # if Kernel k-means
        gram_matrix = datas
        c = initial_c if type(initial_c) != type(None) \
            else RandomCluster(datas.shape[0], k)
    else: # if k-means
        u = initial_u if type(initial_u) != type(None) \
            else RandomMean(k, datas.shape[1])
        c = np.zeros((datas.shape[0], k))
# u(2x2)
# datas(1500x2)
    while(1):
        # E-step
        if not isKernel: # if k-means
            # datapoints, mean
            w = Euclidean(datas, u) # 1500x2
        else:
            w = KernelTrick(gram_matrix, c)

        
        # M-step
        # np.arange(3) -> array([0, 1, 2])
        update_c = np.zeros(w.shape)
        # argmin return the indices
        update_c[np.arange(w.shape[0]), np.argmin(w, axis=1)] = 1 # w.shape[0] = 1500, Rnk-update
        delta_c = np.count_nonzero(np.abs(update_c - c))

        if not isKernel: # k-means
            u = GetMeanFromCluster(datas, update_c) # determination of mean
        else:
            u = None

        yield update_c, u # return Rnk and mean

        if delta_c == 0:
            converge_count-=1
            # find converge
            if converge_count == 0:
                break
        
        c = update_c
    return

# initial setting
data_source = cf.circle
k = 2
kernel = RBFkernel(80)

similarity_matrix = kernel(data_source, data_source)
#cf.showGram(similarity_matrix)
W = similarity_matrix

# degree
D = np.sum(W, axis=1)*np.eye(W.shape[0])
# graph laplacian
L = D - W
eigen_values, eigen_vectors = np.linalg.eig(L)

# generate new vector space to present origin data
# if eigen vector is fully connect , 
# we need to drop it and use next one
# graph laplacian
L = D - W
eigen_values, eigen_vectors = np.linalg.eig(L)

sorted_idx = np.argsort(eigen_values)

U = []
use_eigen_idx = []
current_i = 0
current_k = 0

while current_k < k:
    evc = eigen_vectors[:,sorted_idx[current_i]]
    if True:
    #if (np.var(evc) > 10**-20):
        U.append(evc[:,None])
        use_eigen_idx.append(sorted_idx[current_i])
        current_k += 1
        
    current_i += 1

U = np.concatenate(U, axis=1)

launcher = kmeans(U, k, 
                  initial_u=GetMeanFromCluster(
                      U,
                      RandomCluster(U.shape[0], k)
                  )
                 )
all_spectral_c, all_spectral_u = runKmeans(launcher)
cf.showClustering(data_source, all_spectral_c[-1])
cf.showGram(cf.ReorderGram(W, all_spectral_c[-1]))

# show eigen vector and value
sorted_idx = np.argsort(eigen_values)

show_number = 6

plt.plot(eigen_values[sorted_idx[:show_number]],'.')
plt.xticks(range(show_number), sorted_idx[:show_number])
plt.title('Eigen values')
plt.xlabel('index of eigen vector')
plt.show()

plt.figure(figsize=(4*show_number,3))

for i in range(show_number):
    plt.subplot(1, show_number, i+1)
    plt.plot(eigen_vectors[:,sorted_idx[i]])
    plt.title('Eigen vectors '+str(sorted_idx[i]))

plt.show()