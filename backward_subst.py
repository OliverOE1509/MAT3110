import numpy as np


n = 4
A = np.triu(np.random.randint(1,10, size = (n,n)))
#A = np.array([[6,5], [0,5]])

b = np.random.randint(10, 15, size = n)
#b = np.array([13, 10])

x = np.zeros(n)

#x[n-1] = b[n-1] / A[n-1,n-1]


#print(A)

def back_subst(A: np.ndarray):

	if A.shape[0] != A.shape[1]:
		raise ValueError("Input must be square, douche...")

	for i in range(n-1, -1, -1):
		tmp = b[i]
		for j in range(n-1, i, -1):
			tmp -= x[j] * A[i, j]
			#print(i, j)
		x[i] = tmp / A[i,i]

	return A





#print(b)
#print(x)