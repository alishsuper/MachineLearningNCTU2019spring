import imageio
import numpy as np

import auxiliary_function as af
import kmeans as km

circle, moon = af.load_dataset()

dataset = moon # circle
k = 2 # 3, 4
random_mean = -1 + 2*np.random.random((k, dataset.shape[1]))
initial_mean = km.k_means_plus_plus(dataset, k)
error = 7 # any number

set_classification = km.k_means(dataset, k, initial_mean, error)

img = af.save_cluster(dataset, set_classification, k)
if id(initial_mean) == id(random_mean):
    if id(dataset) == id(moon):
        imageio.mimsave('kmeans_gifs/moon/number_of_clusters_' + str(k) + '_kmeans.gif', img, fps=1)
    else:
        imageio.mimsave('kmeans_gifs/circle/number_of_clusters_' + str(k) + '_kmeans.gif', img, fps=1)
else:
    if id(dataset) == id(moon):
        imageio.mimsave('kmeans++/kmeans_gifs/moon/number_of_clusters_' + str(k) + '_kmeans.gif', img, fps=1)
    else:
        imageio.mimsave('kmeans++/kmeans_gifs/circle/number_of_clusters_' + str(k) + '_kmeans.gif', img, fps=1)
