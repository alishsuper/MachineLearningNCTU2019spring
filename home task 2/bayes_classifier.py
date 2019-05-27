import argparse
import dataset
import numpy as np

#TODO. self.p[][][] move to configuration file
class NaiveBayesian(object):
    """
    Implement the discrete naive bayesian classifier
    Tally the frequency of the values of each pixel into 32 bins
    """
    #Constructor
    def __init__(self, num_features, num_classes, num_bins):
        self.num_features = num_features #784 - 28x28
        self.num_classes = num_classes #10
        self.num_bins = num_bins #32

    def train(self, train_images, train_labels):
        print('[+] Training starts ...')
        
        # Number of training images
        N = train_images.shape[0]
        # Prior probability (uniform), just work with initial training labels
        self.prior = np.array([(train_labels == i).sum()/N for i in range(self.num_classes)], dtype=np.float)

        # Training
        self.p = np.zeros((self.num_classes, self.num_features, self.num_bins))
        f = open("digit_discrete.txt","w+")
        dig_discr = np.array([[0 for row in range(28)] for col in range(28)], dtype=int)
        for i in range(N): # 60000
            if i % 2000 == 0:
                print("[*] Processing %d images" % i)

            for d in range(self.num_features): # 784
                # Tally the frequency of the values of each pixel into 32 bins. For example, The gray
                # level of 0 to 7 should be classified to bin 0, gray level of 8 to 15 should be bin 1 ... etc
                _bin = int(train_images[i][d] // (256/self.num_bins))
                self.p[int(train_labels[i])][d][_bin] += 1

        # imagination of digits
        for c in range(10):
            k = 0
            for i in range(28):
                for j in range(28):
                    if self.p[c][k][0] >= self.prior[c]*40000:
                        dig_discr[i][j] = 0
                    else:
                        dig_discr[i][j] = 1
                    k += 1
            for i in range(28):
                for j in range(28):
                    f.write(str(dig_discr[i][j]))
                    f.write(' ')
                f.write('\n')
            f.write('\n')
        f.close()

        # Calculate the distribution for each pixel
        for i in range(10):
            for d in range(self.num_features): # 784
                for b in range(self.num_bins): # 32
                    if self.p[i][d][b] == 0:
                        self.p[i][d][b] += 0.0001

                    self.p[i][d] /= self.p[i][d].sum()

        print("[-] Training finished ...\n")

    def inference(self, test_images, test_labels, N, print_num=10):
        log_posterior = np.zeros((N, 10))
        predictions = np.zeros(N)

        print('[+] Testing starts ...')

        log_prior = np.log10(self.prior)
        for i in range(N): # 10000
            if i % 1000 == 0:
                print("[*] Test %d images" % i)

            # Calculate the log likelihood given the class
            log_likelihood = np.zeros(len(log_prior))
            for l in range(self.num_classes): # 10
                # Aussme pixels are independent to each other
                for d in range(self.num_features): # 784
                    _bin = int(test_images[i][d] // int(256//self.num_bins))
                    log_likelihood[l] += np.log10(self.p[l][d][_bin])

            log_posterior[i] = log_likelihood + log_prior
            predictions[i] = np.argmax(log_posterior[i])

        print("\n[+] ---------- %d Prediction Results ----------" % print_num)

        N_sum = np.array([[0 for row in range(print_num)] for col in range(print_num)], dtype=float)
        for i in range(print_num):
            for j in range(log_posterior.shape[1]):
                N_sum[0][i] = N_sum[0][i] + log_posterior[i][j]

        for i in range(print_num):
            print('log_posterior')
            for j in range(log_posterior.shape[1]):
                print(j, ':', log_posterior[i][j]/N_sum[0][i])

            print("prediction: %d   true label: %d\n" % (predictions[i], test_labels[i]))

        accuracy = np.sum(predictions==test_labels[:N]) / N

        return predictions, 1 - accuracy

# TODO
class GaussianNaiveBayesian(object):

    def __init__(self, num_features, num_classes):
        self.num_features = num_features # 784
        self.num_classes = num_classes # 10

        self.mean = np.zeros((num_classes, num_features)) # expectation
        self.variance = np.zeros((num_classes, num_features)) # variance

    def train(self, train_images, train_labels):
        print("[+] Training starts ...")

        N = train_images.shape[0] # 60000
        # Prior probability (uniform), just work with initial training labels
        # N_l - number of digits
        N_l = np.array([(train_labels == i).sum() for i in range(self.num_classes)], dtype=np.float)
        self.prior = N_l / N

        f = open("digit_continuous.txt","w+")
        dig_cont = np.array([[0 for row in range(28)] for col in range(28)], dtype=int)
        # Udpate mean of Gaussian
        for c in range(self.num_classes): # 10
            # _sum.shape = (784,) N = 60000
            # summation x[i]
            _sum = np.sum(train_images[n] if train_labels[n] == c else 0.0 for n in range(N))
            # sum(x[i])/n
            self.mean[c] = _sum / N_l[c]

            # imagination
            k = 0
            for i in range(28):
                for j in range(28):
                    if self.mean[c][k] <= 80:
                        dig_cont[i][j] = 0
                    else:
                        dig_cont[i][j] = 1
                    k += 1
            for i in range(28):
                for j in range(28):
                    f.write(str(dig_cont[i][j]))
                    f.write(' ')
                f.write('\n')
            f.write('\n')
        f.close()

        # Update variance of Gaussian
        for c in range(self.num_classes):
            _sum = np.sum((train_images[n] - self.mean[c]) ** 2 if train_labels[n] == c else 0.0 for n in range(N))
            self.variance[c] = _sum / N_l[c]

        print("[-] Training finished ...\n")

    def inference(self, test_images, test_labels, N, print_num=10):
        log_posterior = np.zeros((N, 10))
        predictions = np.zeros(N)

        print('[+] Testing starts ...')

        for i in range(N): # 10000
            if i % 1000 == 0:
                print("[*] Test %d images" % i)

            log_posterior[i] = self.classify(test_images[i])
            predictions[i] = np.argmax(log_posterior[i])

        print("\n[+] ---------- %d Prediction Results ----------" % print_num)

        N_sum = np.array([[0 for row in range(print_num)] for col in range(print_num)], dtype=float)
        for i in range(print_num):
            for j in range(log_posterior.shape[1]):
                N_sum[0][i] = N_sum[0][i] + log_posterior[i][j]

        for i in range(print_num):
            print('log_posterior')
            for j in range(log_posterior.shape[1]):
                print(j, ':', log_posterior[i][j]/N_sum[0][i])

            print("prediction: %d   true label: %d\n" % (predictions[i], test_labels[i]))

        accuracy = np.sum(predictions==test_labels[:N]) / N

        return log_posterior, 1 - accuracy

    def classify(self, image):
        result = [self._log_probability(image, _class) for _class in range(self.num_classes)]
        return np.array(result)

    def _log_probability(self, x, c): # c is a digit (0-9), x is a test image (0-255)
        log_prior_c = np.log10(self.prior[c])
        log_likelihood = 0.0
        for d in range(self.num_features): # 784
            if self.mean[c][d] <= 90:
                self.mean[c][d] = 0
            else:
                self.mean[c][d] = 255

            if self._gaussian(x[d], self.mean[c][d], self.variance[c][d]) == 0:
                log_likelihood += 0
            else:
                log_likelihood += np.log10(self._gaussian( x[d], self.mean[c][d], self.variance[c][d] ))

        return log_prior_c + log_likelihood

    def _gaussian(self, x, u, var):
        if var < 1e-5:
            return 0.0001

        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-np.power((x - u), 2) / (2*var))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='specify the data file', default='C:/Users/ali_2/Desktop/classes/Machine Learning/home task 2/')
    parser.add_argument('--mode', type=int, help='0 for discrete mode, 1 for continuous mode', default=0)
    args = parser.parse_args()

    # Load dataset
    train_images, train_labels, test_images, test_labels = dataset.load_mnist_dataset(args.dir)

    if args.mode == 0:
        # discrete mode
        print('------ Discrete Naive Bayesian Classifier ------')
        model = NaiveBayesian(train_images.shape[1], 10, 32)
    elif args.mode == 1:
        # continuous mode
        print('------ Gaussian Naive Bayesian Classifier ------')
        model = GaussianNaiveBayesian(train_images.shape[1], 10)
    else:
        pass

    model.train(train_images, train_labels)
    predictions, error_rate = model.inference(test_images, test_labels, 10000)
    print("Error Rate: %.4f %%" % (error_rate*100))