import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objs as go

# local libraries
import common_function as cf
import kmeans as km

# initial setting
data_source = cf.circle
k = 2
kernel = km.RBFkernel(80)

similarity_matrix = kernel(data_source, data_source)
# initial Gram matrix
cf.showGram(similarity_matrix)
# weight matrix
W = similarity_matrix

# Diagonal (degree matrix)
D = np.sum(W, axis=1)*np.eye(W.shape[0])
# graph laplacian
L = D - W

# compute eigen values and eigen vectors
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

launcher = km.kmeans(U, k, initial_u=km.GetMeanFromCluster(U, km.RandomCluster(U.shape[0], k)))

all_spectral_c, all_spectral_u = km.runKmeans(launcher)

###############
# type of cluster, dataset, number of clusters, Rnk, mean
cf.AddClusteringProcess(2, data_source, k, all_spectral_c, all_spectral_u)

# save the video
cf.GenerateAllClusteringProcess()
###############

#cf.showClustering(data_source, all_spectral_c[-1])

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
    plt.title('Eigen vectors ' + str(sorted_idx[i]))

plt.show()

gogodata = [go.Scatter(x=U[:,0], y=U[:,1], mode='markers', text=[str(i) in range(U.shape[0])], name='represent data')]
cf.showClustering(U, all_spectral_c[-1])

# reorder to show eigen vector
sorted_idx = np.argsort(eigen_values)
# ???
reorder_data_idx = cf.ReorderByCluster(all_spectral_c[-1])

show_number = 6

plt.plot(eigen_values[sorted_idx[:show_number]],'.')
plt.xticks(range(show_number), sorted_idx[:show_number])
plt.title('Eigen values')
plt.xlabel('index of eigen vector')
plt.show()

plt.figure(figsize=(4*show_number,3))

for i in range(show_number):
    plt.subplot(1, show_number, i+1)
    plt.plot(eigen_vectors[reorder_data_idx,sorted_idx[i]])
    plt.title('Eigen vectors ' + str(sorted_idx[i]))

plt.show()