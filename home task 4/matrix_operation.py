import math

# transpone of the matrix
def matrix_transpose(x):
	transpose = []
	for i in range(len(x[0])):
		temp = []
		for j in range(len(x)):
			temp.append(x[j][i])
		transpose.append(temp)
	return transpose

# matrix multiplication
def matrix_mul(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(B[0])):
			temp.append(0)
		result.append(temp)
	for i in range(len(A)):
		for j in range(len(B[0])):
			for k in range(len(B)):
				result[i][j] += A[i][k] * B[k][j]
	return result

# matrix subtraction
def matrix_minus(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] - B[i][j])
		result.append(temp)
	return result

# creation a diagonal matrix
def I_mul_scalar(scalar, size):
	result = []
	for i in range(size):
		temp = []
		for j in range(size):
			if(i == j):
				temp.append(scalar)
			else:
				temp.append(0)
		result.append(temp)
	return result

# LU decomposition for inverse matrix
def LU_decomposition(A):
	L_inverse = I_mul_scalar(1, len(A))
	for i in range(len(A) - 1):
		L_temp = I_mul_scalar(1, len(A))
		for j in range(i + 1, len(A)):
			L_temp[j][i] = (-1) * A[j][i] / A[i][i]
		A = matrix_mul(L_temp, A)
		L_inverse = matrix_mul(L_temp, L_inverse)
	return L_inverse, A

# get inverse matrix
def matrix_inverse(U, L_inverse):
	result = I_mul_scalar(0, len(U))
	for i in range(len(U)-1, -1, -1):
		for j in range(len(result[0])):
			temp = 0
			for k in range(len(result)):
				temp += U[i][k] * result[k][j]
			result[i][j] = (L_inverse[i][j] - temp) / U[i][i]
	return result

def condition_check(A):
	for i in range (0, len(A)):
		if math.isnan(A[0][0]):
			A[0][0] = 2.0
		if math.isnan(A[1][0]):
			A[1][0] = 5.0
		if math.isnan(A[2][0]):
			A[2][0] = -5.0

# deteminnt of Hessian
def determinant(A):
	return A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])