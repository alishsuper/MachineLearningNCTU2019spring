import numpy as np
import numba

# scan mnist data
@numba.jit
def open_file():
	data_type = np.dtype("int32").newbyteorder('>')
	
	data = np.fromfile("train-images.idx3-ubyte", dtype = "ubyte")
	X = data[4 * data_type.itemsize:].astype("float64").reshape(60000, 28 * 28).transpose()
	X = np.divide(X, 128).astype("int") # divide into two bins

	labels = np.fromfile("train-labels.idx1-ubyte",dtype = "ubyte").astype("int")
	labels = labels[2 * data_type.itemsize : ].reshape(60000)
	return X, labels

# TODO: Prior, Mean, Mean_prev
@numba.jit
def initial_parameters():
	Prior = np.full((10, 1), 0.1, dtype=np.float64)
	Mean = np.random.rand(28 * 28, 10).astype(np.float64) # uniform distribution
	Mean_prev = np.zeros((28 * 28, 10), dtype=np.float64)
	Z = np.full((10, 60000), 0.1, dtype=np.float64)
	return Prior, Mean, Mean_prev, Z

# posterior
@numba.jit
def E_Step(X, Mean, Prior, Z):
	for n in range(0, 60000):
		# make equals to zero
		temp = np.zeros(shape=(10), dtype=np.float64)
		for k in range(0, 10):
			temp1 = np.float64(1.0)
			# likelihood
			for i in range(0, 28 * 28):
				if X[i][n]: # if 1
					temp1 *= Mean[i][k]
				else: # if 0
					temp1 *= (1 - Mean[i][k])
			temp[k] = Prior[k][0] * temp1
		# sum of all probabilities
		temp2 = np.sum(temp)
		# we don't want to get infinity value
		if temp2 == 0:
			temp2 = 1
		for k in range(0, 10):
			# calculate responsibility
			Z[k][n] = temp[k] / temp2
	return Z

# update likelihood and prior
@numba.jit
def M_Step(X, Mean, Prior, Z):
	N = np.sum(Z, axis=1)
	for j in range(0, 28*28):
		for m in range(0, 10):
			# w*x
			temp = np.dot(X[j], Z[m])
			# sum of posteriors
			temp1 = N[m]
			# we don't want to get infinity value
			if temp1 == 0:
				temp1 = 1
			# P0 = w*x/sum(w), expectation
			Mean[j][m] = (temp / temp1)

	for i in range(0, 10):
		# prior
		Prior[i][0] = N[i] / 60000
	return Mean, Prior

@numba.jit
def condition_check(Prior, Mean, Mean_prev, Z, condition):
	temp = 0
	for i in range(0, 10):
		if Prior[i][0] == 0:
			condition = 0
			temp = 1
			temp1 = Mean_prev
			Prior, Mean, temp2, Z = initial_parameters()
			Mean_prev = temp1
			break
	if temp == 0:
		condition += 1
	return Prior, Mean, Mean_prev, Z, condition

# difference between expectations
@numba.jit
def difference(Mean, Mean_prev):
	temp = 0
	for i in range(0, 28 * 28):
		for j in range(0, 10):
			temp += abs(Mean[i][j] - Mean_prev[i][j])
	return temp

@numba.jit
def print_Mean(Mean):
	Mean_new = Mean.transpose()
	for i in range(0, 10):
		print("\nclass: ", i)
		for j in range(0, 28 * 28):
			if j % 28 == 0 and j != 0:
				print("")
			if Mean_new[i][j] >= 0.5:
				print("1", end=" ")
			else:
				print("0", end=" ")
		print("")

@numba.jit
def decide_label(X, labels, Mean, Prior):
	table = np.zeros(shape=(10, 10), dtype=np.int)
	relation = np.full((10), -1, dtype=np.int)
	for n in range(0, 60000):
		temp = np.zeros(shape=10, dtype=np.float64)
		for k in range(0, 10):
			temp1 = np.float64(1.0)
			# likelihood
			for i in range(0, 28 * 28):
				if X[n][i] == 1: # if 1
					temp1 *= Mean[i][k]
				else: # if 0
					temp1 *= (1 - Mean[i][k])
			temp[k] = Prior[k][0] * temp1
		table[labels[n]][np.argmax(temp)] += 1

	for i in range(1, 11):
		ind = np.unravel_index(np.argmax(table, axis=None), table.shape)
		print(ind)
		relation[ind[0]] = ind[1]
		for j in range(0, 10):
			table[ind[0]][j] = -1 * i
			table[j][ind[1]] = -1 * i
	return relation

@numba.jit
def print_labeled_class(Mean, relation):
	Mean_new = Mean.transpose()

	for i in range(0, 10):
		print("\nlabeled class: ", i)
		label = relation[i]
		for j in range(0, 28 * 28):
			if j % 28 == 0 and j != 0:
				print("")
			if Mean_new[label][j] >= 0.5:
				print("1", end=" ")
			else:
				print("0", end=" ")
		print("")

@numba.jit
def print_confusion_matrix(X, labels, Mean, Prior, relation):
	error = 60000
	confusion_matrix = np.zeros(shape=(10,2,2), dtype=np.int)
	for n in range(0, 60000):
		temp = np.zeros(shape=10, dtype=np.float64)
		for k in range(0, 10):
			temp1 = np.float64(1.0)
			for i in range(0, 28 * 28):
				if X[n][i] == 1:
					temp1 *= Mean[i][k]
				else:
					temp1 *= (1 - Mean[i][k])
			temp[k] = Prior[k][0] * temp1
		predict = np.argmax(temp)
		for i in range (0, 10):
			if relation[i] == predict:
				predict = i
				break
		for k in range(0, 10):
			if labels[n] == k:
				if predict == k:
					confusion_matrix[k][0][0] += 1 #TP
				else:
					confusion_matrix[k][0][1] += 1 # FP
			else:
				if predict == k:
					confusion_matrix[k][1][0] += 1 # FN
				else:
					confusion_matrix[k][1][1] += 1 # TN
	
	for i in range(0, 10):
		print("\n---------------------------------------------------------------\n")
		print("Confusion Matrix {}: ".format(i))
		print("\t\tPredict number {}\t Predict not number {}".format(i, i))
		print("Is number {}\t\t{}\t\t\t{}".format(i, confusion_matrix[i][0][0], confusion_matrix[i][0][1]))
		print("Isn't number {}\t\t{}\t\t\t{}\n".format(i, confusion_matrix[i][1][0], confusion_matrix[i][1][1]))
		print("Sensitivity (Successfully predict number {})\t: {}".format(i, confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])))
		print("Specificity (Successfully predict not number {})\t: {}".format(i, confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])))
	
	for i in range(0, 10):
		error -= confusion_matrix[i][0][0]
	return error

if __name__ == "__main__":

	X, labels = open_file()

	Prior, Mean, Mean_prev, Z = initial_parameters()

	iteration = 0
	condition = 0
	while(True):

		iteration += 1
		# E-step:
		Z = E_Step(X, Mean, Prior, Z)

		# M-step:
		Mean, Prior = M_Step(X, Mean, Prior, Z)

		Prior, Mean, Mean_prev, Z, condition = condition_check(Prior, Mean, Mean_prev, Z, condition)

		diff = difference(Mean, Mean_prev)

		if diff < 10 and np.sum(Prior) > 0.95 and condition >= 10:
			break
		Mean_prev = Mean
		print_Mean(Mean)
		print("No. of Iteration: {}, Difference: {}\n".format(iteration, diff))
		print("---------------------------------------------------------------\n")
	
	print("---------------------------------------------------------------\n")
	relation = decide_label(X.transpose(), labels, Mean, Prior)
	print('rel', relation)
	print_labeled_class(Mean, relation)
	error = print_confusion_matrix(X.transpose(), labels, Mean, Prior, relation)
	print("\nTotal iteration to converge: {}".format(iteration))
	print("Total error rate: {}".format(error/60000.0))