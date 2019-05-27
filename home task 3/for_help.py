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

# used last post variance (identity matrix)
# original x
x_or = np.array([ [1, -0.64152, 0.4115479104, -0.26401621548] ], dtype=float)
# Sn
#identity matrix
c_id = np.identity(len(sn_v))
# transponse x
xt_v = np.array([ [1], [-0.64152], [0.4115479104], [-0.26401621548] ], dtype=float)
#print(xt_v, sn_v)
c_id = np.identity(len(sn_v))
c_var0 = np.array(np.dot(x_or, c_id))
c_var1 = np.array(np.dot(c_var0, xt_v))
print('Multiplication predict variance True (-0.64152, 0.19039)')
print(1 + c_var1)

# inverse Sn
sn = 1/(1 + c_var1)
# transponse x
xt_v = np.array([ [1], [-0.64152], [0.4115479104], [-0.26401621548] ], dtype=float)
m_v = sn * xt_v * 0.19039
print('Multiplication post mean (-0.64152, 0.19039)')
print(m_v)

# used last post mean
b_x = np.array([ [1], [0.07122], [0.0050722884], [0.00036124837] ], dtype=float)
c_x = np.array(np.dot(m_v.T, b_x))
print('Multiplication predict mean True (0.07122, 1.63175)')
print(c_x)

# used last post mean
a_x = np.array([ [0.9107496675, 1.9265499885, 3.1119297129, 4.1312375189] ], dtype=float)
b_x = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)
c_x = np.array(np.dot(a_x, b_x))
print('Multiplication predict mean True (0.36500, 2.22705)')
print(c_x)

# used last post variance
# original x
x_or = np.array([ [1, 0.07122, 0.0050722884, 0.00036124837] ], dtype=float)
# Sn
sn_v = np.array([ [0.6227289276,   0.2420256620,  -0.1552634839,   0.0996041049], 
[0.2420256620,   0.8447365161,   0.0996041049,  -0.0638976884], 
[-0.1552634839,   0.0996041049,   0.9361023116,   0.0409914289], 
[0.0996041049,  -0.0638976884,   0.0409914289,   0.9737033172] ], dtype=float)
# transponse x
xt_v = np.array([ [1], [0.07122], [0.0050722884], [0.00036124837] ], dtype=float)
#print(xt_v, sn_v)
c_id = np.identity(len(sn_v))
c_var0 = np.array(np.dot(x_or, sn_v))
c_var1 = np.array(np.dot(c_var0, xt_v))
print('Multiplication predict variance True (0.07122, 1.63175)')
print(1 + c_var1)

# transponse x
a_v = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)
# original x
b_v = np.array([ [1, 0.365, 0.133225, 0.048627] ], dtype=float)
c_v = np.array(np.dot(a_v, b_v))
# prior variance
sn_v = np.array([ [0.0051883836,  -0.0004416700,  -0.0086000319,   0.0008247001], 
[-0.0004416700,   0.0401966605,   0.0012708906,  -0.0554822477], 
[-0.0086000319,   0.0012708906,   0.0265353911,  -0.0031205875], 
[0.0008247001,  -0.0554822477,  -0.0031205875,   0.0937197255] ], dtype=float)
#identity matrix
c_id = np.identity(len(c_v))
bI = np.array(np.dot(lu_decomposition_inverse(sn_v), c_id))
c_res = bI + c_v
print('Multiplication post variance True (0.36500, 2.22705)')
print(lu_decomposition_inverse(c_res))

# prior mean
m0_v = np.array([ [0.9107496675], [1.9265499885], [3.1119297129], [4.1312375189] ], dtype=float)
# prior variance
s0_v = np.array([ [0.0051883836,  -0.0004416700,  -0.0086000319,   0.0008247001], 
[-0.0004416700,   0.0401966605,   0.0012708906,  -0.0554822477], 
[-0.0086000319,   0.0012708906,   0.0265353911,  -0.0031205875], 
[0.0008247001,  -0.0554822477,  -0.0031205875,   0.0937197255] ], dtype=float)
# posterior variance
sn_v = np.array([ [0.0051731092, -0.0004872471, -0.0085815201, 0.0008842340], 
[-0.0004872471, 0.0400606628, 0.0013261280, -0.0553046044], 
[-0.0085815201, 0.0013261280, 0.0265129556, -0.0031927398], 
[0.0008842340, -0.0553046044, -0.0031927398, 0.0934876838] ], dtype=float)
# transponse x
xt_v = np.array([ [1], [0.365], [0.133225], [0.048627] ], dtype=float)
xty_v = xt_v * 2.22705
s0_m0 = np.array(np.dot(lu_decomposition_inverse(s0_v), m0_v))
c_v = s0_m0 + xty_v
m_v = np.array(np.dot(sn_v, c_v))
print('Multiplication post mean True (0.36500, 2.22705)')
print(m_v)

# transponse x
a_v = np.array([ [1], [-0.64152], [0.4115479104], [-0.26401621548] ], dtype=float)
# original x
b_v = np.array([ [1, -0.64152, 0.4115479104, -0.26401621548] ], dtype=float)
c_v = np.array(np.dot(a_v, b_v))
# prior variance
sn_v = 1
#identity matrix
c_id = np.identity(len(c_v))
bI = c_id
c_res = bI + c_v
print('Multiplication post variance True (-0.64152, 0.19039)')
print(lu_decomposition_inverse(c_res))

# prior mean
#m0_v = np.array([ [1], [2], [3], [4] ], dtype=float)
# posterior variance
sn_v = np.array([ [0.6227289276,   0.2420256620,  -0.1552634839,   0.0996041049], 
[0.2420256620,   0.8447365161,   0.0996041049,  -0.0638976884], 
[-0.1552634839,   0.0996041049,   0.9361023116,   0.0409914289], 
[0.0996041049,  -0.0638976884,   0.0409914289,   0.9737033172] ], dtype=float)
# prior variance
#c_id = np.identity(len(sn_v))
#s0_m0 = np.array(np.dot(c_id, m0_v))
# transponse x
xt_v = np.array([ [1], [-0.64152], [0.4115479104], [-0.26401621548] ], dtype=float)
print(xt_v.shape)
xt_y = xt_v * 0.19039
m_v = np.array(np.dot(sn_v, xt_y))
print('Multiplication post mean True (-0.64152, 0.19039)')
print(m_v)