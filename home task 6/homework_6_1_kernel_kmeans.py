import imageio

# local libraries
import auxiliary_function as af
import kernelkmeans as kkm

circle, moon = af.load_dataset()

dataset = moon # circle
k = 2 # 3, 4
change_gamma = False # True
gamma = 15 # any number
error = 7 # any number

gram_matrix = kkm.RBF(dataset, dataset, gamma)

set_classification = kkm.kernel_k_means(gram_matrix, k, error)

img = af.save_cluster(dataset, set_classification, k)

if (change_gamma):
    if id(dataset) == id(moon):
        imageio.mimsave('change_gamma/moon/number_of_clusters_' + str(k) + '_gamma_' + str(gamma) + '_kernel-kmeans.gif', img, fps=1)
    else:
        imageio.mimsave('change_gamma/circle/number_of_clusters_' + str(k) + '_gamma_' + str(gamma) + '_kernel-kmeans.gif', img, fps=1)
else:
    if id(dataset) == id(moon):
        imageio.mimsave('kernel-kmeans_gifs/moon/number_of_clusters_' + str(k) + '_kernel-kmeans.gif', img, fps=1)
    else:
        imageio.mimsave('kernel-kmeans_gifs/circle/number_of_clusters_' + str(k) + '_kernel-kmeans.gif', img, fps=1)