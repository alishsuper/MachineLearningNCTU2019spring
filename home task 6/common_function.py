# tool and common function
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

# Load datasets
circle_df = pd.read_csv('circle.txt', names=['x','y'])
circle = np.array(circle_df)
moon_df = pd.read_csv('moon.txt', names=['x','y'])
moon = np.array(moon_df)

__data_spaces = {'circle':circle, 'moon':moon}
__cluster_spaces = {'kmeans':0,'kernel-kmeans':1,'spectral':2}
__fig_frames_cache = {}
__fig_frames_dirty = {}

def showClustering(datas, C, u=None, title='', k=None, color_k = ['blue', 'darkorange', 'green', 'deeppink']):

    if k == None:
        k = len(C[0])
    
    frame = []

    plt.figure(figsize=(10,10))
    
    frame = __showClustering(datas, C, u, k, color_k)
    
    plt.title(title)
    plt.legend()
    plt.show()
    #plt.savefig(str(id(C)) + '.png')


# dataset, Rnk, mean, cluster
def __showClustering(datas, C, u, k, color_k = ['blue', 'darkorange', 'green', 'deeppink']):
    frame = []
    for i in range(k):
        
        if type(u) != type(None):
            # add mean
            l, = plt.plot(u[i][0], u[i][1], 'o', c=color_k[i], markersize=50.0, alpha=0.3)
            frame.append(l)
            # add center to the mean
            l, = plt.plot(u[i][0], u[i][1], '*', c=color_k[i], alpha=0.3, label='clustering {} mean'.format(i+1))
            frame.append(l)
        clustering_datas = datas[C[:,i]==1,:]
        # cluster datapoints
        l, = plt.plot(clustering_datas[:,0], clustering_datas[:,1], '.', c=color_k[i], label='clustering '+str(i+1))
        frame.append(l)

    return frame

def ReorderByCluster(c):
    new_order = np.array([])
    for i in range(c.shape[1]):
        new_order = np.append(new_order, np.where(c[:,i]==1)[0])
    new_order = new_order.astype('int32')
    return new_order
    
def ReorderGram(gram, c):
    new_order = ReorderByCluster(c)
    return gram[new_order,:][:,new_order]

def showGram(gram):
    plt.imshow(gram, cmap='gray')
    plt.show()

def __default_title(idx, c, u):
    return 'iter [{}]'.format(idx)

def __clusteringsave(filename, fig, frames):
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True, repeat_delay=500)
    
    writer = FFMpegWriter(fps=2, metadata=dict(artist='TonyLee'), bitrate=1800)
    ani.save(filename + ".mp4", writer=writer)

def GenerateClusteringProcess(filename, datas, all_c, all_u, title_func=__default_title):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()
    
    frames = []
    
    for idx, (c, u) in enumerate(zip(all_c, all_u)):
        kk = c.shape[1]
        # dataset, Rnk, mean, 
        f = __showClustering(datas, c, u, kk)

        title = plt.text(0.5, 1.01, title_func(idx, c, u), horizontalalignment='center', 
                         verticalalignment='bottom', transform=ax.transAxes)
    
        f += [title]
        frames.append(f)
    if type(filename) == type(None):
        return fig, frames
    plt.legend()

    #__clusteringsave(filename, fig, frames)
    
    #fig.clear(keep_observers=True)

# type of cluster, dataset, number of clusters, Rnk, mean
def AddClusteringProcess(cluster_type, data_type, k, all_c, all_u, init_type='random'):
    if type(data_type) != type(''):
        for key, value in __data_spaces.items():

            if id(value) == id(data_type):
                datas = data_type
                data_type = key
                break
    else:
        datas = __data_spaces[data_type]
    if type(cluster_type) != type(''):
        for key, value in __cluster_spaces.items():
            if value == cluster_type:
                cluster_type = key
                break
    else:
        if not cluster_type in __cluster_spaces:
            raise AttributeError
    
    process_title = '{}_data={}_k={}_init={}'.format(cluster_type, data_type, k, init_type)
    
    fig_frames = list(GenerateClusteringProcess(None, datas, all_c, all_u))
    #fig_frames[0].clear(keep_observers=True)
    
    __fig_frames_cache[process_title] = fig_frames
    __fig_frames_dirty[process_title] = True
    
def GenerateAllClusteringProcess():
    for filename, v in __fig_frames_cache.items():
        if __fig_frames_dirty[filename]:
            __clusteringsave(filename, v[0], v[1])
            __fig_frames_dirty[filename] = False