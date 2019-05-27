import os
import numpy as np
import struct

def load_mnist_dataset(_dir):
    train_images = np.reshape(read_idx3_data(os.path.join(_dir, 'train-images.idx3-ubyte')), (-1, 28*28))
    train_labels = read_idx1_data(os.path.join(_dir, 'train-labels.idx1-ubyte'))
    test_images  = np.reshape(read_idx3_data(os.path.join(_dir, 't10k-images.idx3-ubyte')), (-1, 28*28))
    test_labels  = read_idx1_data(os.path.join(_dir, 't10k-labels.idx1-ubyte'))

    return train_images, train_labels, test_images, test_labels

def read_idx3_data(filename):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    """
    bin_data = open(filename, 'rb').read()

    offset = 0
    magic_number, num_images, num_rows, num_cols = struct.unpack_from('>iiii', bin_data, offset) # unpack from the buffer according to the format string fmt
    print('magic number:\t%d\ntotal images:\t%d\nimage size:\t%d*%d' % (magic_number, num_images, num_rows, num_cols))

    offset += struct.calcsize('>iiii')
    image_size = num_rows * num_cols
    fmt_image = '>' + str(image_size) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images

def read_idx1_data(filename):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    """

    bin_data = open(filename, 'rb').read()

    offset = 0
    magic_number, num_images = struct.unpack_from('>ii', bin_data, offset)
    # print('magic number:\t%d\ntotal images:\t%d\n' % (magic_number, num_images))

    offset += struct.calcsize('>ii')

    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from('>B', bin_data, offset)[0]
        offset += struct.calcsize('>B')

    return labels