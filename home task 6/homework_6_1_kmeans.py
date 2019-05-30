import common_function as cf
import kmeans as km

# set dataset and number of cluster
data_source = cf.circle
k = 2

# run python generator
launcher = km.kmeans(data_source, k)

# get Rnk and mean
all_kmean_c, all_kmean_u = km.runKmeans(launcher)

# type of cluster, dataset, number of clusters, Rnk, mean
cf.AddClusteringProcess(0, data_source, k, all_kmean_c, all_kmean_u)

# save the video
cf.GenerateAllClusteringProcess()


