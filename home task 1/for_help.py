import numpy as np

#matrix multiplication
res1 = np.array([ [0.3765992302,   0.1254838660,  -0.1000441911,   0.0627881634], 
[0.1254838660,   0.7895542671,   0.1257503020,  -0.0813299447], 
[-0.1000441911,   0.1257503020,   0.9237138418,   0.0492510997], 
[0.0627881634,  -0.0813299447,   0.0492510997,   0.9681964094] ], dtype=float)

res_x = np.array([ [1], [0.07122], [0.005072], [0.000361] ], dtype=float)
res_mult = np.array(np.dot(res1, res_x))
coef = 1.63175
print('Multiplication')
print(res_mult*coef)

a_x = np.array([ [0.9107404583, 1.9265225090, 3.1119408740, 4.1312734131] ], dtype=float)
b_x = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)
print('ab', a_x.shape, b_x.shape)
c_x = np.array(np.dot(a_x, b_x))
print('Multiplication predict mean')
print(c_x)

# POSTERIOR VARIANCE
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
        C = np.identity(len(res))

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

# transponse x
a_v = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)
# original x
b_v = np.array([ [1, 0.365, 0.133225, 0.048627] ], dtype=float)
c_v = np.array(np.dot(a_v, b_v))
#identity matrix
c_id = np.identity(len(c_v))
c_res = c_id + c_v
print('Multiplication post variance')
print(c_res)

# Sn
sn_v = np.array([ [0.0051731092, -0.0004872471, -0.0085815201, 0.0008842340], 
[-0.0004872471, 0.0400606628, 0.0013261280, -0.0553046044], 
[-0.0085815201, 0.0013261280, 0.0265129556, -0.0031927398], 
[0.0008842340, -0.0553046044, -0.0031927398, 0.0934876838] ], dtype=float)
# transponse x
xt_v = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)

m_v = (0.1) * np.array(np.dot(sn_v, xt_v)) * 2.22705
#m_v = np.array((0.1)*np.dot(sn_v, xt_v)) * 2.22705
#c_res = c_id + (-0.1)*c_v
#print(a_v.shape)
print('Multiplication post mean')
print(m_v)

# original x
x_or = np.array([ [1, 0.365, 0.133225, 0.048627] ], dtype=float)
# Sn
sn_v = np.array([ [0.0051731092, -0.0004872471, -0.0085815201, 0.0008842340], 
[-0.0004872471, 0.0400606628, 0.0013261280, -0.0553046044], 
[-0.0085815201, 0.0013261280, 0.0265129556, -0.0031927398], 
[0.0008842340, -0.0553046044, -0.0031927398, 0.0934876838] ], dtype=float)
# transponse x
xt_v = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)
#print(xt_v, sn_v)
c_var0 = np.array(np.dot(x_or, sn_v))
c_var1 = np.array(np.dot(c_var0, xt_v))
print('Multiplication predict variance')
print(1 + c_var1)

# original x
x_or = np.array([ [1, -0.76990, 0.59274601, -0.45635515309] ], dtype=float)
# Sn
sn_v = np.array([ [0.0051883836,  -0.0004416700,  -0.0086000319,   0.0008247001], 
[-0.0004416700,   0.0401966605,   0.0012708906,  -0.0554822477], 
[-0.0086000319,   0.0012708906,   0.0265353911,  -0.0031205875], 
[0.0008247001,  -0.0554822477,  -0.0031205875,   0.0937197255] ], dtype=float)
# transponse x
xt_v = np.array([ [1], [-0.76990], [0.59274601], [-0.45635515309] ], dtype=float)
#print(xt_v, sn_v)
c_var0 = np.array(np.dot(x_or, sn_v))
c_var1 = np.array(np.dot(c_var0, xt_v))
print('Multiplication predict variance')
print(1 + c_var1)
