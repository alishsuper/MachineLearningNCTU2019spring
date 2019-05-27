import random
import math
import argparse

class GenerateRandomNumber(object):

    #Constructor
    def __init__(self, hasSpare, spare):
        self.hasSpare = hasSpare #false
        self.spare = spare #0

    def marsaglia_polar(self, mean, variance):

        if self.hasSpare:
            print("y")
            self.hasSpare = False
            return mean + variance * self.spare

        self.hasSpare = True
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            s = x * x + y * y
            if s < 1:
                print("x")
                t = math.sqrt((-2) * math.log(s)/s)
                self.spare = y * t
                return mean + variance * x * t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=float, help='mean', default=3.0)
    parser.add_argument('--variance', type=float, help='variance', default=5.0)
    args = parser.parse_args()

    random_value_class = GenerateRandomNumber(False, 0)

    income_numbers = 200

    for c in range(income_numbers):
        rand_value = random_value_class.marsaglia_polar(args.mean, args.variance)
        print(rand_value)
