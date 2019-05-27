import csv
import numpy as np
from svmutil import svm_train, svm_predict

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

# training
kernel_function = ["Linear", "Polynomial", "RBF"]
for j in range(0, 3):
	print("Kernel Function: {}".format(kernel_function[j]))
	kernel_parameter = "-q -t " + str(j)
	kernel_model = svm_train(y_train, x_train, kernel_parameter)
	pred_labels, pred_acc, pred_values = svm_predict(y_test, x_test, kernel_model)