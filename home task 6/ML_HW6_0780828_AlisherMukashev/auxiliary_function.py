from matplotlib import pyplot as plt
import numpy as np

def load_dataset():
    circle = np.genfromtxt('circle.txt', delimiter=',')
    moon = np.genfromtxt('moon.txt', delimiter=',')
    return circle, moon

def save_cluster(dataset, set_classification, k):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()
    colors = ['g', 'y', 'r', 'b']

    frames = []
    img = []
    for iteration, classif in enumerate(set_classification):
        for i in range(k):
            clustering_dataset = dataset[classif[:, i] == 1, :]
            l = ax.scatter(clustering_dataset[:, 0], clustering_dataset[:, 1], c=colors[i])
            frames.append(l)

        ax.title.set_text('Step: {}'.format(iteration+1))
    
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img.append(image)
        ax.clear()

    return img
    plt.legend()