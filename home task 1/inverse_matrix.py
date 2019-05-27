import numpy as np

#identity matrix
C = np.identity(4)

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

#getting inverse matrix
res_newt = np.array([ [25, 5, 1, 1], [64, 8 , 1, 1], [144, 12, 1, 1], [169, 13, 1, 1] ], dtype=float)
X_Newt = np.array([[0 for row in range(len(res_newt))] for col in range(len(res_newt))], dtype=float)
X_Newt = lu_decomposition_inverse(res_newt)
print(X_Newt)