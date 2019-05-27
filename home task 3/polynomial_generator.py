import random
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

class GenerateRandomNumber(object):

    #Constructor
    def __init__(self, hasSpare, spare):
        self.hasSpare = hasSpare #false
        self.spare = spare #0

    def marsaglia_polar(self, mean, variance):

        if self.hasSpare:
            self.hasSpare = False
            return mean + variance * self.spare

        self.hasSpare = True
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            s = x * x + y * y
            if (s < 1 and s > 0):
                t = math.sqrt((-2) * math.log(s)/s)
                self.spare = y * t
                return mean + variance * x * t

    def generate_y(self, basis_number, w_vector, uniform_x):
        y = 0
        for i in range(basis_number):
            y = y + w_vector[0][i] * (uniform_x ** i)
        return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, help='number of basis', default=4)
    parser.add_argument('--a', type=float, help='noise precision parameter', default=1)
    parser.add_argument('--w', type=float, nargs="+", help='weights', default=[1,2,3,4])
    args = parser.parse_args()

    random_value_class = GenerateRandomNumber(False, 0)

# Input --n=4 --a=1 --w '1' '2' '3' '4'
    w_vector = np.array([args.w], dtype=float)
    noise_var_a = args.a
    basis_number = args.n
    income_numbers = 200

    y_number = np.array([[0 for row in range(income_numbers)] for col in range(1)], dtype=float)
    uniform_x = np.array([[0 for row in range(income_numbers)] for col in range(1)], dtype=float)
    for c in range(income_numbers):
        rand_value = random_value_class.marsaglia_polar(0, noise_var_a)
        uniform_x[0][c] = random.uniform(-1, 1)
        y_number[0][c] = random_value_class.generate_y(basis_number, w_vector, uniform_x[0][c]) + rand_value

# ground truth
    x_axis = np.arange(-2, 2, 0.01)
    y_mean = random_value_class.generate_y(basis_number, w_vector, x_axis)
    y1_variance = y_mean + noise_var_a
    y2_variance = y_mean - noise_var_a

    plt.plot(uniform_x[0], y_number[0], 'bo')
    plt.plot(x_axis, y_mean, 'k')
    plt.plot(x_axis, y1_variance, 'r')
    plt.plot(x_axis, y2_variance, 'r')
    plt.show()