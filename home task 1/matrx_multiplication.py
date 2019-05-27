import numpy as np

res = np.array([ [25, 5, 1], [64, 8 , 1], [144, 12, 1] ], dtype=float)
C = np.identity(len(res))
res_newt = np.array(np.dot(res, C))

print(res_newt)