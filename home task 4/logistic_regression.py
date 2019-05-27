import numpy as np
import math
import matplotlib.pyplot as plt
import matrix_operation as mo
import argparse
import random

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

# for gradient descent
def update(A, B):
	result = []
	rate = 0.01
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] - B[i][j] * rate)
		result.append(temp)
	return result

def get_data(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
	c1x = []
	c1y = []
	c2x = []
	c2y = []
	X = []
	y = []
	for i in range(0, n):
		temp1 = []
		x1 = random_value_class.marsaglia_polar(mx1, math.sqrt(vx1))
		y1 = random_value_class.marsaglia_polar(my1, math.sqrt(vy1))
		c1x.append(x1)
		c1y.append(y1)
		temp1.append(x1)
		temp1.append(y1)
		temp1.append(1.0)
		X.append(temp1)
		y.append([0.0])
		
		temp2 = []
		x2 = random_value_class.marsaglia_polar(mx2, math.sqrt(vx2))
		y2 = random_value_class.marsaglia_polar(my2, math.sqrt(vy2))
		c2x.append(x2)
		c2y.append(y2)
		temp2.append(x2)
		temp2.append(y2)
		temp2.append(1.0)
		X.append(temp2)
		y.append([1.0])
	return X, y, c1x, c1y, c2x, c2y

# sigmoid function
def sigmoid(A):
	matrix = []
	for i in range(0, len(A)):
		temp = []
		for j in range(0, len(A[0])):
			temp.append(1.0 / (1.0 + np.exp(-1.0 * A[i][j])))
		matrix.append(temp)
	return matrix

# difference between current and previous data for interrupting
def difference(A, B):
	temp = True
	for i in range(0, len(A)):
		if abs(A[i][0] - B[i][0]) > 0.05:
			temp = False
			break
	return temp

# for plotting graph
def draw(X, c1x, c1y, c2x, c2y, gradient_w, y, newton_w):
	plt.subplot(131)
	plt.title("Ground Truth")
	plt.scatter(c1x, c1y, c = 'r')
	plt.scatter(c2x, c2y, c = 'b')

	print("Gradient Descent:\n")
	confusion_matrix = [[0, 0], [0, 0]]
	predict = sigmoid(mo.matrix_mul(X, gradient_w))
	c1x = []
	c1y = []
	c2x = []
	c2y = []
	# points for graph
	for i in range(0, len(predict)):
		if predict[i][0] < 0.5:
			c1x.append(X[i][0])
			c1y.append(X[i][1])
		else:
			c2x.append(X[i][0])
			c2y.append(X[i][1])

	# confusion matrix
	for i in range(0, len(predict)):
		# is cluster 1 (actual yes)
		if y[i][0] == 0:
			# predict cluster 1
			if predict[i][0] < 0.5:
				confusion_matrix[0][0] += 1 # TP
			# predict cluster 2
			else:
				confusion_matrix[0][1] += 1 # FP

		# is cluster 2 (actual no)
		if y[i][0] == 1:
			# predict cluster 1
			if predict[i][0] < 0.5:
				confusion_matrix[1][0] += 1 # FN
			# predict cluster 2
			else:
				confusion_matrix[1][1] += 1 # TN

	print("w:")
	for i in range(0, len(gradient_w)):
		print(gradient_w[i][0])
	print("\nconfusion_matrix")
	print("\t\t Predict cluster 1 Predict cluster 2")
	print("Is cluster 1\t\t {}\t\t{}" .format(confusion_matrix[0][0], confusion_matrix[0][1]))
	print("Is cluster 2\t\t {}\t\t{}\n" .format(confusion_matrix[1][0], confusion_matrix[1][1]))
	print("Sensitivity (Successfully predict cluster 1): {}" .format(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])))
	print("Specificity (Successfully predict cluster 2): {}" .format(confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])))

	plt.subplot(132)
	plt.title("Gradient Descent")
	plt.scatter(c1x, c1y, c = 'r')
	plt.scatter(c2x, c2y, c = 'b')

	print("\n------------------------------------\nNewton's Method:\n")
	confusion_matrix = [[0, 0], [0, 0]]
	predict = sigmoid(mo.matrix_mul(X, newton_w))
	c1x = []
	c1y = []
	c2x = []
	c2y = []
	for i in range(0, len(predict)):
		if predict[i][0] < 0.5:
			c1x.append(X[i][0])
			c1y.append(X[i][1])
		else:
			c2x.append(X[i][0])
			c2y.append(X[i][1])
	for i in range(0, len(predict)):
		if y[i][0] == 0:
			if predict[i][0] < 0.5:
				confusion_matrix[0][0] += 1
			else:
				confusion_matrix[0][1] += 1
		if y[i][0] == 1:
			if predict[i][0] < 0.5:
				confusion_matrix[1][0] += 1
			else:
				confusion_matrix[1][1] += 1

	print("w:")
	for i in range(0, len(newton_w)):
		print(newton_w[i][0])
	print("\nconfusion_matrix")
	print("\t\t Predict cluster 1 Predict cluster 2")
	print("Is cluster 1\t\t {}\t\t{}" .format(confusion_matrix[0][0], confusion_matrix[0][1]))
	print("Is cluster 2\t\t {}\t\t{}\n" .format(confusion_matrix[1][0], confusion_matrix[1][1]))
	print("Sensitivity (Successfully predict cluster 1): {}" .format(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])))
	print("Specificity (Successfully predict cluster 2): {}" .format(confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])))

	plt.subplot(133)
	plt.title("Newton's Method")
	plt.scatter(c1x, c1y, c = 'r')
	plt.scatter(c2x, c2y, c = 'b')

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":

	# scan input data
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=int, help='number of data points', default=50)

	parser.add_argument('--mx1', type=float, help='mean', default=1)
	parser.add_argument('--my1', type=float, help='mean', default=1)
	parser.add_argument('--mx2', type=float, help='mean', default=10)
	parser.add_argument('--my2', type=float, help='mean', default=10)

	parser.add_argument('--vx1', type=float, help='variance', default=2)
	parser.add_argument('--vy1', type=float, help='variance', default=2)
	parser.add_argument('--vx2', type=float, help='variance', default=2)
	parser.add_argument('--vy2', type=float, help='variance', default=2)

	args = parser.parse_args()

	random_value_class = GenerateRandomNumber(False, 0)

# Input example --n=50 --mx1=1 --my1=1 --mx2=3 --my2=3 --vx1=2 --vy1=2 --vx2=4 --vy2=4
	X, y, c1x, c1y, c2x, c2y = get_data(args.n, args.mx1, args.vx1, args.my1, args.vy1, args.mx2, args.vx2, args.my2, args.vy2)
	
	# gradient descent
	w = [[0.0], [0.0], [0.0]] # previous one, 3x1
	new_w = [[0.0], [0.0], [0.0]] # current one, 3x1

	# transponse of matrix X
	X_transpose = mo.matrix_transpose(X) # 3x50

	while(True):
		# x*W
		sigmoid_input = mo.matrix_mul(X, w)
		# X_T*(f(W_T*x)-Y)
		partial_derivative = mo.matrix_mul(X_transpose, mo.matrix_minus(sigmoid(sigmoid_input), y))
		# lambda = 0.1
		new_w = update(w, partial_derivative)
		if difference(new_w, w):
			break
		w = new_w
	gradient_w = w

	# newton's method
	w = [[0.0], [0.0], [0.0]]
	new_w = [[0.0], [0.0], [0.0]]
	while(True):
		# diagonal matrix
		D = []
		for i in range(0, len(X)):
			temp = []
			for j in range(0, len(X)):
				if i == j:
					# -X*w
					temp1 = -1.0 * (X[i][0] * w[0][0] + X[i][1] * w[1][0] + X[i][2] * w[2][0])
					temp2 = np.exp(temp1)
					if math.isinf(temp2):
						temp2 = np.exp(700)
					temp.append(temp2 / ((1 + temp2) ** 2))
				else:
					temp.append(0.0)
			D.append(temp)
		# A_T*D*A - Hessian matrix
		Hessian = mo.matrix_mul(X_transpose, mo.matrix_mul(D, X))
		sigmoid_input = mo.matrix_mul(X, w)
		# X_T*(sigm-y)
		partial_derivative = mo.matrix_mul(X_transpose, mo.matrix_minus(sigmoid(sigmoid_input), y))
		if mo.determinant(Hessian) == 0:
			# Gradient Descent
			new_w = update(w, partial_derivative)
		else:
			# Newton's Method
			L_inverse, U = mo.LU_decomposition(Hessian)
			Hessian_inverse = mo.matrix_inverse(U, L_inverse)
			new_w = mo.matrix_minus(w, mo.matrix_mul(Hessian_inverse, partial_derivative))
		mo.condition_check(new_w)
		if difference(new_w, w):
			break
		w = new_w
draw(X, c1x, c1y, c2x, c2y, gradient_w, y, new_w)