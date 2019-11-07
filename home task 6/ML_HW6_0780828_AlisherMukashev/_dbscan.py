import matplotlib.pyplot as plt
import numpy as np

def dbscan(dataset, min_points, eps):
    Y = np.zeros(dataset.shape[0])
    eps = eps**2
    C = 1
    step=0
    
    def pick_next():
        while True:
            i = np.nonzero(Y==0)[0]
            if i.size == 0:
                break
            else:
                yield i[0]
        pass
    
    def scan(i):
        n = np.sum((dataset - dataset[i])**2, axis=1) <= eps
        if sum(n) >= min_points:
            n[i] = False
            n = np.logical_and(n, np.logical_or(Y==0, Y==-1))
            Y[n] = C
            return np.where(n)[0]
        else:
            Y[n] = -1
            return np.ndarray(0)
        pass
    
    for i in pick_next():
        F = [i]
        for i in F:
            F.extend(scan(i).tolist())
            step += 1
            yield Y, dataset[i], step
        C += 1
    pass

def plot_dbscan(ax, dataset, Y, x):
    ax.scatter(dataset[:,0], dataset[:,1], c=Y)
    ax.scatter(x[0], x[1], s=100, c='r', marker='x')

def rundbscan(dataset, min_points, eps):
    img = []
    fig = plt.figure(figsize=(10,10))
    axes = plt.axes()

    for Y, x, step in dbscan(dataset, min_points, eps):
        axes.clear()
        plot_dbscan(axes, dataset, Y, x)
        axes.title.set_text("Step: {}".format(step))
        
        if (step % 50) == 0 or step == 1:

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img.append(image)
        plt.show(block=False)
        plt.pause(0.01)

    if (step % 50) != 0:
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img.append(image)
    return img