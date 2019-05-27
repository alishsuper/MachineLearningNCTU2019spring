import sys
import numpy as np
import matplotlib.pyplot as plt

#scan array from txt file
file_name = sys.argv[1]
def file_lengthy(fname):
        with open(fname) as f:
                for i, l in enumerate(f):
                        pass
        return i + 1

#amount of x and y
m = file_lengthy(file_name) - 1

#scan data points
x, y = np.loadtxt(file_name, delimiter=',', usecols=(0,1), unpack=True, max_rows=m)
#scan number of bases and lambda
n, l = np.loadtxt(file_name, dtype=int, delimiter=',', skiprows=m, usecols=(0,1), unpack=True)

#design matrix
i1 = np.ones(m)[np.newaxis]
x1 = x[np.newaxis]
xx = np.concatenate((i1.T, x1.T), 1)
j = 2
while j<n:
    xx = np.concatenate((xx, (x1 ** j).T), 1)    
    j = j + 1

#matrix multiplication and summation, preparing matrix for inverse
res_newt = np.array(np.dot(xx.T, xx))

#identity matrix
C = np.identity(len(x))

#matrix summation with lambda for LSE
lambda_identity = np.array([[0 for row in range(len(res_newt))] for col in range(len(res_newt))], dtype=float)
res_lse = np.array([[0 for row in range(len(res_newt))] for col in range(len(res_newt))], dtype=float)

for i in range(0, len(res_newt)):
    for j in range(0, len(res_newt)):
        lambda_identity[i][j] = C[i][j] * l
        res_lse[i][j] = res_newt[i][j] + lambda_identity[i][j]

#lu decomposition and inverse of matrix
def lu_decomposition_inverse(res):
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

#help to visualization
def get_curve(coeff, x_p):
        max = 0
        min = 0
        for i in range(0, len(x_p)):
                if x[i] > max:
                        max = x_p[i]
                if x[i] < min:
                        min = x_p[i]

        x_axis = np.arange(min, max, 0.01)

        curve = coeff[0]
        for i in range(1, len(coeff)):
                curve = curve + coeff[i]*(x_axis ** i)

        return x_axis, curve

#help to get value of total error
def get_curve_for_error(coeff, x_p):
        curve = coeff[0]
        for i in range(1, len(coeff)):
                curve = curve + coeff[i]*(x_p ** i)

        return curve

#print out
print('n =', n)
print('lambda = ', l, '\n')

#------------------------#
#NEWTON'S METHOD
#------------------------#
print('Newton"s method:')

#NEW METHOD
loops = 20
epsilon = 0.0001
kol = 0
coef = np.array([0 for col in range(len(res_newt))], dtype=float)[np.newaxis]
y1 = y[np.newaxis]

while(True):
        err = np.array(np.dot(np.array(np.dot(xx.T, xx)), coef.T)) - np.array(np.dot(xx.T, y1.T))
        if abs(np.sum(err)) < epsilon or kol > loops:
                break
        else:
                coef1 = coef - (np.array(np.dot(np.array(lu_decomposition_inverse(res_newt)), err))).T
        kol = kol + 1
        coef = coef1

#getting inverse matrix
#X_Newt = np.array([[0 for row in range(len(res_newt))] for col in range(len(res_newt))], dtype=float)
#X_Newt = lu_decomposition_inverse(res_newt)

#matrix multiplication, getting coefficients and the best line
coeff_newt = coef[0].T#np.array(np.dot(np.array(np.dot(X_Newt, xx.T)), y))
str_newt = repr(coeff_newt[0])
for i in range(1, len(coeff_newt)):
        str_newt = str_newt + '+' + repr(coeff_newt[i]) + '*X^' + repr(i) 
print('Fitting line:', str_newt)

#total error        
curve_newt_for_error = get_curve_for_error(coeff_newt, x)
total_error_newt = np.array(np.dot((curve_newt_for_error-y).T, (curve_newt_for_error-y)))
print('Total error:', total_error_newt, '\n')

#visualisation
ax1 = plt.subplot(211)
ax1.set_title('Newton"s method')
ax1.plot(x, y, 'ro')
x_axis_newt, curve_newt = get_curve(coeff_newt, x)
ax1.plot(x_axis_newt, curve_newt)
plt.subplots_adjust(hspace=0.4)

#------------------------#
#LSE METHOD
#------------------------#
print('LSE:')
#getting inverse matrix
X_LSE = np.array([[0 for row in range(len(res_lse))] for col in range(len(res_lse))], dtype=float)
X_LSE = lu_decomposition_inverse(res_lse)

#matrix multiplication, getting coefficients and the bset line
coeff_lse = np.array(np.dot(np.array(np.dot(X_LSE, xx.T)), y))
str_lse = repr(coeff_lse[0])
for i in range(1, len(coeff_lse)):
        str_lse = str_lse + '+' + repr(coeff_lse[i]) + '*X^' + repr(i) 
print('Fitting line:', str_lse)

#total error
curve_lse_for_error = get_curve_for_error(coeff_lse, x)
total_error_lse = np.array(np.dot((curve_lse_for_error-y).T, (curve_lse_for_error-y)))
print('Total error:', total_error_lse)

#visualisation
ax2 = plt.subplot(212)
ax2.set_title('LSE')
ax2.plot(x, y, 'ro')
x_axis_lse, curve_lse = get_curve(coeff_lse, x)
ax2.plot(x_axis_lse, curve_lse)

#show two subplots
plt.show()