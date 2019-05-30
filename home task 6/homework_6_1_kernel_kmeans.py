# local libraries
import common_function as cf
import kmeans as km

# initial setting
data_source = cf.circle
k = 2

kernel = km.RBFkernel(100)

# calculate Gram matrix
gram_matrix = kernel(data_source, data_source)

launcher = km.kmeans(gram_matrix, k, isKernel=True)

all_kkmean_c, all_kkmean_u = km.runKmeans(launcher)

cf.AddClusteringProcess(1, data_source, k, all_kkmean_c, all_kkmean_u)

cf.GenerateAllClusteringProcess()