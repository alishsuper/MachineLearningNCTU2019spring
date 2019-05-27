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

class Helpers(object):

    #lu decomposition and inverse of matrix
    def lu_decomposition_inverse(self, res):
        L = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)
        U = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)

        #lu decomposition
        for j in range(len(res)):
                U[0][j] = res[0][j]
                L[j][0] = res[j][0]/U[0][0]

        for i in range(1, len(res)):
                for j in range(i, len(res)):
                        s1 = sum((L[i][k1]*U[k1][j]) for k1 in range(0, i))
                        U[i][j] = res[i][j] - s1

                        s2 = sum(L[j][k2]*U[k2][i] for k2 in range(i))
                        L[j][i] = (res[j][i] - s2)/U[i][i]

        #inverse of matrix
        Z = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)
        X = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)
        C = np.identity(len(res))

        #forward substitution
        for i in range(0, len(res)):
                for j in range(0, len(res)):
                        s1 = sum((L[i][k1]*Z[k1][j]) for k1 in range(0, i))
                        Z[i][j] = C[i][j] - s1

        #back substituion
        for i in range((len(res)-1), -1, -1):
                for j in range(0, len(res)):
                        s1 = sum((U[i][k1]*X[k1][j]) for k1 in range((i+1), len(res)))
                        X[i][j] = (Z[i][j] - s1)/U[i][i]
        
        return X

    def design_matrix(self, x, n):
        D = np.array([[0 for row in range(n)] for col in range(1)], dtype=float)
        D[0] = 1
        for j in range(1, n):
            D[0][j] = x ** j
        return D

class BayesianLinearRegression(object):

    def get_post_variance(self, x, n, a, pr_v, is_it_first):
        # original x: x, n
        xd_v = helpers.design_matrix(x, n)
        # transponse x
        xt_v = xd_v.T
        # a*X.T*X
        noise_a = a
        c1_v = noise_a * np.array(np.dot(xt_v, xd_v))
        # prior variance
        prior_v = np.array([[0 for row in range(len(c1_v))] for col in range(len(c1_v))], dtype=float)
        #identity matrix
        I_m = np.identity(len(c1_v))
        # prior variance
        if is_it_first:
            # b*I
            b_I = pr_v * I_m
        else:
            prior_v = pr_v
            # b*I
            b_I = np.array(np.dot(helpers.lu_decomposition_inverse(prior_v), I_m))
        # a*X.T*X + b*I
        post_v_inv = c1_v + b_I
        return post_v_inv
    
    def get_post_mean(self, post_v_inv, x, n, a, y, pr_v, prior_m, is_it_first):
        # posterior variance
        post_var = helpers.lu_decomposition_inverse(post_v_inv)
        # prior variance
        # original x: x, n
        xd_v = helpers.design_matrix(x, n)
        # transponse x
        xt_v = xd_v.T
        # a*x.T*y
        y_m = y
        noise_a = a
        xt_y = noise_a * xt_v * y_m
        # identity matrix
        I_m = np.identity(len(post_var))
        # prior_var*prior_mean
        # prior variance
        if is_it_first:
            # b*I
            prior_var = pr_v * I_m
        else:
            # b*I
            prior_var = np.array(np.dot(pr_v, I_m))
        # prior mean
        prior_mean = prior_m
        pr_v_pr_m = np.array(np.dot(helpers.lu_decomposition_inverse(prior_var), prior_mean))
        # post_var*(prior_var*prior_mean + a*x.T*y)
        c1v = pr_v_pr_m + xt_y
        post_mean = np.array(np.dot(post_var, c1v))
        return post_mean

    def get_predict_mean(self, prior_m, x, n):
        # original x: x, n, transponse because of format
        xd_v = helpers.design_matrix(x, n).T
        predict_mean = np.array(np.dot(prior_m.T, xd_v))
        return predict_mean

    def get_predict_variance(self, prior_v, x, n, a):
        # original x: x, n, because of format
        xd_v = helpers.design_matrix(x, n).T
        # prior variance
        prior_var = prior_v
        # transponse x
        xt_v = xd_v.T
        # x.T*prior_v*x
        c1_v = np.array(np.dot(xt_v, prior_var))
        c2_v = np.array(np.dot(c1_v, xd_v))
        # 1/a + x.T*prior_v*x
        predict_var = 1/a + c2_v
        return predict_var

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, help='prior variance', default=1)
    parser.add_argument('--n', type=int, help='number of basis', default=4)
    parser.add_argument('--a', type=float, help='noise precision parameter', default=1)
    parser.add_argument('--w', type=float, nargs="+", help='weights', default=[1,2,3,4])
    args = parser.parse_args()

    random_value_class = GenerateRandomNumber(False, 0)

    helpers = Helpers()

    bayes_lin_regr = BayesianLinearRegression()

# Input --b=1 --n=4 --a=1 --w '1' '2' '3' '4'
    w_vector = np.array([args.w], dtype=float)
    noise_var_a = args.a
    basis_number = args.n
    prior_variance = args.b
    prior_mean = np.zeros(shape=(basis_number, 1))
    post_m = 0
    post_v = 0

    y_ten = np.array([[0 for row in range(10)] for col in range(1)], dtype=float)
    x_ten = np.array([[0 for row in range(10)] for col in range(1)], dtype=float)
    w_vector_ten = np.array([[0 for row in range(1)] for col in range(basis_number)], dtype=float)

    y_fifty = np.array([[0 for row in range(50)] for col in range(1)], dtype=float)
    x_fifty = np.array([[0 for row in range(50)] for col in range(1)], dtype=float)
    w_vector_fifty = np.array([[0 for row in range(1)] for col in range(basis_number)], dtype=float)

    x_final = []
    y_final = []

    c = 0
    while True:
        rand_value = random_value_class.marsaglia_polar(0, noise_var_a)
        current_x = random.uniform(-1, 1)
        current_y = random_value_class.generate_y(basis_number, w_vector, current_x) + rand_value
        x_final.append(current_x)
        y_final.append(current_y)

        print('Add data point', current_x, current_y)

        # for visualisation, after 10 incomes
        if c < 10:
            x_ten[0][c] = current_x
            y_ten[0][c] = current_y
            w_vector_ten = post_m
            prior_variance_ten = post_v

        # for visualisation, after 50 incomes
        if c < 50:
            x_fifty[0][c] = current_x
            y_fifty[0][c] = current_y
            w_vector_fifty = post_m
            prior_variance_fifty = post_v

        if c == 0:
            is_it_first = True
        else:
            is_it_first = False
            prior_variance = post_v
            prior_mean = post_m

        c += 1

        # Posterior variance
        # x, n, a, prior_var, is_it_first
        post_v_inv = bayes_lin_regr.get_post_variance(current_x, basis_number, noise_var_a, prior_variance, is_it_first)
        post_v = helpers.lu_decomposition_inverse(post_v_inv)

        # Posterior mean
        # post_v_inv, x, n, y, a, prior_v, prior_m
        post_m = bayes_lin_regr.get_post_mean(post_v_inv, current_x, basis_number, current_y, noise_var_a, prior_variance, prior_mean, is_it_first)
        print('Posterior mean:')
        print(post_m)

        print('Posterior variance:')
        print(post_v)

        # Prediction mean
        # prior_m, x, n
        predict_m = bayes_lin_regr.get_predict_mean(prior_mean, current_x, basis_number)

        # Prediction variance
        # prior_v, x, n, a
        predict_v = bayes_lin_regr.get_predict_variance(prior_variance, current_x, basis_number, noise_var_a)

        print('Predictive dictribution ~ N(', predict_m[0][0], ',', predict_v[0][0], ')')

        summ = 0
        for j in range(basis_number):
            if abs(abs(post_m[j])-abs(w_vector[0][j])) < 0.2:
                summ +=1

        if summ == basis_number:
            break

    print('Number of iterations =', c)

# VISUALISATION
# ground truth
    x_axis = np.arange(-2, 2, 0.01)
    # black line
    y_mean = random_value_class.generate_y(basis_number, w_vector, x_axis)
    # red lines
    y1_variance = y_mean + noise_var_a
    y2_variance = y_mean - noise_var_a

    ax1 = plt.subplot(221)
    ax1.set_title('Ground truth')
    ax1.plot(x_axis, y_mean, 'k')
    ax1.plot(x_axis, y1_variance, 'r')
    ax1.plot(x_axis, y2_variance, 'r')
    plt.subplots_adjust(hspace=0.4)

# Predict result
    # black line
    y_mean_final = random_value_class.generate_y(basis_number, post_m.T, x_axis)
    predict_v_final = np.array([[0 for row in range(len(x_axis))] for col in range(1)], dtype=float)
    for k in range(len(x_axis)):
        predict_v_final[0][k] = bayes_lin_regr.get_predict_variance(prior_variance, x_axis[k], basis_number, noise_var_a)

    # red lines
    y1_variance_final = y_mean_final + predict_v_final[0]
    y2_variance_final = y_mean_final - predict_v_final[0]

    ax2 = plt.subplot(222)
    ax2.set_title('Predict result')
    ax2.plot(x_axis, y_mean_final, 'k')
    ax2.plot(x_axis, y1_variance_final, 'r')
    ax2.plot(x_axis, y2_variance_final, 'r')
    ax2.plot(x_final, y_final, 'bo')
    plt.subplots_adjust(hspace=0.4)

# After 10 incomes
    # black line
    y_mean_ten = random_value_class.generate_y(basis_number, w_vector_ten.T, x_axis)
    predict_v_ten = np.array([[0 for row in range(len(x_axis))] for col in range(1)], dtype=float)
    for k in range(len(x_axis)):
        predict_v_ten[0][k] = bayes_lin_regr.get_predict_variance(prior_variance_ten, x_axis[k], basis_number, noise_var_a)

    # red lines
    y1_variance_ten = y_mean_ten + predict_v_ten[0]
    y2_variance_ten = y_mean_ten - predict_v_ten[0]

    ax3 = plt.subplot(223)
    ax3.set_title('After 10 incomes')
    ax3.plot(x_axis, y_mean_ten, 'k')
    ax3.plot(x_axis, y1_variance_ten, 'r')
    ax3.plot(x_axis, y2_variance_ten, 'r')
    ax3.plot(x_ten, y_ten, 'bo')
    plt.subplots_adjust(hspace=0.4)

# After 50 incomes
    # black line
    y_mean_fifty = random_value_class.generate_y(basis_number, w_vector_fifty.T, x_axis)
    predict_v_fifty = np.array([[0 for row in range(len(x_axis))] for col in range(1)], dtype=float)
    for k in range(len(x_axis)):
        predict_v_fifty[0][k] = bayes_lin_regr.get_predict_variance(prior_variance_fifty, x_axis[k], basis_number, noise_var_a)

    # red lines
    y1_variance_fifty = y_mean_fifty + predict_v_fifty[0]
    y2_variance_fifty = y_mean_fifty - predict_v_fifty[0]

    ax4 = plt.subplot(224)
    ax4.set_title('After 50 incomes')
    ax4.plot(x_axis, y_mean_fifty, 'k')
    ax4.plot(x_axis, y1_variance_fifty, 'r')
    ax4.plot(x_axis, y2_variance_fifty, 'r')
    ax4.plot(x_fifty, y_fifty, 'bo')

    plt.show()