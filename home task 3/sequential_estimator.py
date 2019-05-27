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
            self.hasSpare = False
            return mean + variance * self.spare

        self.hasSpare = True
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            s = x * x + y * y
            if s < 1:
                t = math.sqrt((-2) * math.log(s)/s)
                self.spare = y * t
                return mean + variance * x * t

class SequentialEstimator(object):

    def get_meanvalue(self, last_mean, current_number, current_x):
        return (last_mean + (current_x - last_mean)/current_number)

    def get_variance(self, last_variance, last_mean, current_number, current_x, current_mean):
        return (last_variance + ((current_x - last_mean)*(current_x - current_mean) - last_variance)/current_number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean', type=float, help='mean', default=3.0)
    parser.add_argument('--variance', type=float, help='variance', default=5.0)
    args = parser.parse_args()

    random_value_class = GenerateRandomNumber(False, 0)
    sequent_estim = SequentialEstimator()

    current_mean = random_value_class.marsaglia_polar(args.mean, args.variance)
    current_variance = random_value_class.marsaglia_polar(args.mean, args.variance)

    print('Data point source function: N(', args.mean, ",", args.variance, ')')

    c = 0
    while True:
        last_mean = current_mean
        last_variance = current_variance
        current_x = random_value_class.marsaglia_polar(args.mean, args.variance)
        current_mean = sequent_estim.get_meanvalue(last_mean, c+3, current_x)
        current_variance = sequent_estim.get_variance(last_variance, last_mean, c+3, current_x, current_mean)
        print('Add data point:', current_x)
        print('Mean =', current_mean, '   Variance =', math.sqrt(abs(current_variance)))

        c += 1

        if (abs(abs(current_mean) - abs(last_mean)) < 0.001) and (abs(math.sqrt(abs(current_variance)) - math.sqrt(abs(last_variance))) < 0.001):
            break

    print('Number of iterations =', c)

