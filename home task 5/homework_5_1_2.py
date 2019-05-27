import csv
import numpy as np
from svmutil import svm_train
import numba

@numba.jit
def grid_search(x_train, y_train, x_test, y_test):
	cost = ["1", "2", "3"]
	gamma = ["0.2", "0.3", "0.4"]
	degree = ["2", "3", "4"]
	coef0 = ["0", "1", "2"]
	best_kernel_parameter = ""
	best_accuracy = 0.0
	for i in range(0, 3): # changing kernel function
		for j in range(0, 3): # changing cost (the parameter C of C-SVC)
			kernel_parameter = "-v 3 -q -t " + str(i) + " -c " + cost[j]
			# linear kernel
			if i == 0:
				print("Kernel Function: Linear")
				accuracy = svm_train(y_train, x_train, kernel_parameter)
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_kernel_parameter = kernel_parameter
			# polynomial kernel
			if i == 1:
				for k in range(0, 3): # changing gamma
					for p in range(0, 3): # changing degree
						for q in range(0, 3): # changing coef0
							new_parameter = kernel_parameter + " -g " + gamma[k] + " -d " + degree[p] + " -r " + coef0[q]
							print("Kernel Function: Polynomial")
							accuracy = svm_train(y_train, x_train, new_parameter)
							if accuracy > best_accuracy:
								best_accuracy = accuracy
								best_kernel_parameter = new_parameter
			# RBF kernel
			if i == 2:
				for k in range(0, 3): # changing gamma
					new_parameter = kernel_parameter + " -g " + gamma[k]
					print("Kernel Function: RBF")
					accuracy = svm_train(y_train, x_train, new_parameter)
					if accuracy > best_accuracy:
						best_accuracy = accuracy
						best_kernel_parameter = new_parameter
	print("Best Accuracy: {}".format(best_accuracy))
	print("Best Kernel Parameter: {}".format(best_kernel_parameter))

# read files
with open('X_train.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	x_train = list(csv_reader)
	x_train = [[float(y) for y in x] for x in x_train]
with open('X_test.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	x_test = list(csv_reader)
	x_test = [[float(y) for y in x] for x in x_test]
with open('Y_train.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	temp = list(csv_reader)
	y_train = [y for x in temp for y in x]
	y_train = [int(x) for x in y_train]
with open('Y_test.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	temp = list(csv_reader)
	y_test = [y for x in temp for y in x]
	y_test = [int(x) for x in y_test]

grid_search(x_train, y_train, x_test, y_test)