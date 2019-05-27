import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import numba
from svmutil import svm_train, svm_problem, svm_parameter, svm_predict

@numba.jit
def linear_RBF(u, v, gamma):
	linear_kernel = np.dot(u, v.T)
	rbf_kernel = np.sum(u**2, axis=1)[:, None] + np.sum(v**2, axis=1)[None, :] - 2*linear_kernel
	rbf_kernel = np.abs(rbf_kernel) * -gamma
	rbf_kernel = np.exp(rbf_kernel)
	X = linear_kernel + rbf_kernel
	
	index = np.arange(1, X.shape[0]+1)[:, np.newaxis]
	X = np.c_[index, X]
	X = [list(row) for row in X]
	return X

def arrange_subplots():
	cols = int(np.floor(np.sqrt(4)))
	rows = int(np.ceil(4/cols))
	fig, axes = plt.subplots(cols,rows)
	if not isinstance(axes, np.ndarray):
		axes = np.array([axes])
	return axes

# read csv files
Plot_X = np.genfromtxt('Plot_X.csv', delimiter=',')
Plot_Y = np.genfromtxt('Plot_Y.csv', delimiter=',')

# set hyperparameters
svm_params = ['-q -t 0', '-q -t 1', '-q -t 2', '-q -t 4']
axes = arrange_subplots()
axes = axes.flatten()

for i in range(0, 4):
	ax = axes[i]
	ax.scatter(Plot_X[:,0], Plot_X[:,1], s=1, c='gray')
plt.show(block=False)
plt.pause(0.1)

titles = ("Linear", "Polynomial", "RBF", "Sigmoid", "Linear+RBF")

#Training
for i in range(0, 4):
	ax = axes[i]
	param = svm_parameter(svm_params[i])
	ax.title.set_text(titles[param.kernel_type])
	
	problem = None
	if param.kernel_type == 4:
		problem = svm_problem(Plot_Y, linear_RBF(Plot_X, Plot_X, 0.01), isKernel=True)
	else:
		problem = svm_problem(Plot_Y, Plot_X)
	
	Kernel_model = svm_train(problem, param)
	
	if param.kernel_type == 4:
		Kernel = linear_RBF(Plot_X, Plot_X, 0.001)
	else:
		Kernel = Plot_X

	print("Kernel Function: ", titles[param.kernel_type])
	pred_labels, pred_acc, pred_values = svm_predict(Plot_Y, Kernel, Kernel_model)
	pred_labels = np.array(pred_labels)
	pred_values = np.array(pred_values)
	
	i = np.array(Kernel_model.get_sv_indices()) - 1
	ax.scatter(Plot_X[:,0], Plot_X[:,1], s=1, c=pred_labels)
	ax.scatter(Plot_X[i,0], Plot_X[i,1], s=3, c='r', marker='s')
	plt.show(block=False)
	plt.pause(0.1)

plt.show()