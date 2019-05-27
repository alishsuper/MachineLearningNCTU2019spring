import csv
import numpy as np
from svmutil import svm_train, svm_predict, svm_problem, svm_parameter
import numba

@numba.jit
def linear_RBF(u, v, gamma):
	linear_kernel = np.dot(u, v.T)
	rbf_kernel = np.sum(u**2, axis=1)[:, None] + np.sum(v**2, axis=1)[None, :] - 2*linear_kernel
	rbf_kernel = np.abs(rbf_kernel) * -gamma
	rbf_kernel = np.exp(rbf_kernel)
	X = linear_kernel + rbf_kernel
	
	index = np.arange(1, X.shape[0]+1)[:, np.newaxis]
	X = np.c_[index, X]
	X = [list(row) for row in X]
	return X

# read csv files
x_train = np.genfromtxt('X_train.csv', delimiter=',')
x_test = np.genfromtxt('X_test.csv', delimiter=',')
y_train = np.genfromtxt('Y_train.csv', delimiter=',')
y_test = np.genfromtxt('Y_test.csv', delimiter=',')

# training
print('Linear + RBF')
problem = svm_problem(y_train, linear_RBF(x_train, x_train, 0.01), isKernel=True)
Kernel = linear_RBF(x_test, x_train, 0.01)

parameter = svm_parameter("-q -t 4")
Kernel_model = svm_train(problem, parameter)
pred_labels, pred_acc, pred_values = svm_predict(y_test, Kernel, Kernel_model)