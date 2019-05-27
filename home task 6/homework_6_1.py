import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import random
import math
import IPython.display as IDisplay

# Load datasets
circle_df = pd.read_csv('circle.txt', names=['x','y'])
circle = np.array(circle_df)
moon_df = pd.read_csv('moon.txt', names=['x','y'])
moon = np.array(moon_df)

plt.plot(circle_df['x'], circle_df['y'],'.',label='circle')
plt.legend()
plt.show()
plt.plot(moon_df['x'], moon_df['y'],'.', c='darkorange',label='moon')
plt.legend()
plt.show()