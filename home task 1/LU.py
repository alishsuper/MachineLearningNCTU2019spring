import numpy as np

#res = np.array([ [25, 5, 1, 1], [64, 8 , 1, 1], [144, 12, 1, 1], [169, 13, 1, 1] ], dtype=float)
#res = np.array([ [4, 3], [6, 3] ], dtype=float)
res = np.array([ [0.3765992302,   0.1254838660,  -0.1000441911,   0.0627881634], 
[0.1254838660,   0.7895542671,   0.1257503020,  -0.0813299447], 
[-0.1000441911,   0.1257503020,   0.9237138418,   0.0492510997], 
[0.0627881634,  -0.0813299447,   0.0492510997,   0.9681964094] ], dtype=float)

#lu decomposition
L = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)
U = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)

for j in range(len(res)):
    U[0][j] = res[0][j]
    L[j][0] = res[j][0]/U[0][0]

for i in range(1, len(res)):
    for j in range(i, len(res)):
        s1 = sum((L[i][k1]*U[k1][j]) for k1 in range(0, i))
        U[i][j] = res[i][j] - s1

        s2 = sum(L[j][k2]*U[k2][i] for k2 in range(i))
        L[j][i] = (res[j][i] - s2)/U[i][i]

print('L')
print(L)
print('U')
print(U)

#inverse of matrix
Z = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)
X = np.array([[0 for row in range(len(res))] for col in range(len(res))], dtype=float)
C = np.identity(len(res))

#forward substitution
for i in range(0, len(res)):
    for j in range(0, len(res)):
        s1 = sum((L[i][k1]*Z[k1][j]) for k1 in range(0, i))
        Z[i][j] = C[i][j] - s1

print('Z')
print(Z)

#back substituion
for i in range((len(res)-1), -1, -1):
    for j in range(0, len(res)):
        s1 = sum((U[i][k1]*X[k1][j]) for k1 in range((i+1), len(res)))
        X[i][j] = (Z[i][j] - s1)/U[i][i]

print('X')
print(X)