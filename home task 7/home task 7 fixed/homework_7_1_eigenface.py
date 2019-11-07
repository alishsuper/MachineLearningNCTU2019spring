import numpy as np
from matplotlib import pyplot as plt
import PIL.Image

import auxiliary_function as af

att_faces = []
path = './att_faces/'
for i in range(1,41):
    for j in range(1,11):
            image = PIL.Image.open(path+'s{}/{}.pgm'.format(i, j))
            att_faces += [np.array(image).reshape(1,-1)]
att_faces = np.concatenate(att_faces, axis = 0)
face_eigen = af.calculate_eigenvalue(np.cov(att_faces.T), 25)

for i in range(5):
    temp = np.concatenate([np.concatenate(face_eigen.T.reshape(-1, 112, 92)[5*i:5*(i+1)], axis=1)], axis=0)
    plt.title('first 25 eigenfaces')
    plt.imshow(temp, cmap='gray')
    plt.show()

rand_set = np.random.randint(low=0, high=40, size=10)
rand_set = np.sort(rand_set)
temp = [att_faces[rand_set[i]*10:(rand_set[i]+1)*10] for i in range(10)]
rand_images = np.concatenate(temp, axis=0)

for i in range(10):
    temp = np.concatenate([np.concatenate(rand_images.reshape(-1,112,92)[10*i:10*(i+1)], axis=1)], axis=0)
    plt.title('10 images')
    plt.imshow(temp, cmap='gray')
    plt.show()

for i in range(10):
    temp = np.concatenate([np.concatenate(np.dot(np.dot(rand_images, face_eigen), face_eigen.T).reshape(-1,112,92)[10*i:10*(i+1)], axis=1)], axis=0)
    plt.title('Reconstructed images')
    plt.imshow(temp, cmap='gray')
    plt.show()

