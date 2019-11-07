import numpy as np
from matplotlib import pyplot as plt
import imageio

import auxiliary_function as af
import kmeans as km
import spectral as sp

circle, moon = af.load_dataset()

dataset = circle # moon
k = 2 # 3, 4
gamma = 30 # any number
error = 7 # any number

U = sp.get_eigen_space(dataset, k, gamma)

initial_mean = None # km.k_means_plus_plus(U, k)

if type(initial_mean) == type(None):
    classification = np.zeros((U.shape[0], k))
    ind_min = np.random.randint(k, size=U.shape[0])
    classification[np.arange(U.shape[0]), ind_min] = 1
    mean = np.matmul(classification.T, U) / np.sum(classification, axis=0)[:, None]
else:
    mean = initial_mean

set_classification = km.k_means(U, k, mean, error)

img = af.save_cluster(dataset, set_classification, k)

if type(initial_mean) == type(None):
    if id(dataset) == id(moon):
        imageio.mimsave('spectral-clustering_gifs/moon/number_of_clusters_' + str(k) + '_spectral-clustering.gif', img, fps=1)
    else:
        imageio.mimsave('spectral-clustering_gifs/circle/number_of_clusters_' + str(k) + '_spectral-clustering.gif', img, fps=1)
else:
    if id(dataset) == id(moon):
        imageio.mimsave('kmeans++/spectral-clustering_gifs/moon/number_of_clusters_' + str(k) + '_spectral-clustering.gif', img, fps=1)
    else:
        imageio.mimsave('kmeans++/spectral-clustering_gifs/circle/number_of_clusters_' + str(k) + '_spectral-clustering.gif', img, fps=1)

if k == 2:
    sp.show_eigen_space(k, U, ([row[0] for row in set_classification[k-1]]))