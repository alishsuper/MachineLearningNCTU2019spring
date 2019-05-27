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

m = file_lengthy(file_name) - 1

x = np.loadtxt(file_name, dtype=str, unpack=True, max_rows=m)
#scan number of bases and lambda
a, b = np.loadtxt(file_name, dtype=int, delimiter=',', skiprows=m, usecols=(0,1), unpack=True)

def factorial(var):
        if var == 0:
                return 1
        fact = 1
        for i in range(2, var+1):
                fact = fact * i
        return fact

def fact_matrix(h, t):
        res = factorial(h+t)/factorial(t)/factorial(h)
        return res

def bearnoulli(x, h, t):
        res = np.power(x, h) * np.power((1 - x), t)
        return res
        
for i in range(1, (len(x)+1)):
        print('case', i, ':', x[i-1])
        h = 0
        t = 0
        for j in range(len(x[i-1])):
                if x[i-1][j] == '1':
                        h += 1
                else:
                        t += 1
                pr = h/(h+t)
        likelihood = fact_matrix(h, t) * bearnoulli(pr, h, t)
        print('Likelihood :', likelihood)
        print('Beta prior : a = ', a, 'b = ', b)
        a = a + h
        b = b + t
        print('Beta posterior : a = ', a, 'b = ', b)