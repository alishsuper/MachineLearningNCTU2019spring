import matplotlib.pyplot as plt
import imageio
import numpy as np

import _dbscan
import auxiliary_function as af

circle, moon = af.load_dataset()

dataset = circle # moon
Y = np.zeros(1500)
min_points = 3 # any number
eps = 0.1 # any number

img = _dbscan.rundbscan(dataset, min_points, eps)
imageio.mimsave('dbscan_gifs/dbscan.gif', img, fps=1)

plt.show()