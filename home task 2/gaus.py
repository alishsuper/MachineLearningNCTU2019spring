import numpy as np

var = 0.1
x = 10
u = 0.0003192338387869114

print((1 / np.sqrt(2 * np.pi * var)) * np.exp(-np.power((x - u), 2) / (2*var)))
print(np.power((x - u), 2)/ (2*var))

log_posterior = np.array([ [25, 5, 1, 1], [64, 8 , 1, 1], [144, 12, 1, 1], [169, 13, 1, 10] ], dtype=float)
N = np.array([[0 for row in range(4)] for col in range(4)], dtype=float)
for i in range(4):
    for j in range(log_posterior.shape[1]):
      N[0][i] = N[0][i] + log_posterior[i][j]

for i in range(4):
    print('log_posterior')
    for j in range(log_posterior.shape[1]):
        print(j, ':', log_posterior[i][j]/N[0][i])
    print('\n')

"""
res = np.array([ [1, 1, 1, 1], [0, 0 , 1, 0], [0, 1, 0, 0], [1, 0, 0, 0] ], dtype=int)
print(res)
def main():
     f= open("digit.txt","w+")
     for i in range(len(res)):
        for j in range(len(res)):
            f.write(str(res[i][j]))
            f.write(' ')
        f.write('\n')
     f.close()   
if __name__== "__main__":
  main()
"""