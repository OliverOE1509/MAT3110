import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

n = 30
start = -2
stop = 2
x = np.linspace(start, stop, n)
eps = 1
np.random.seed(1)
r = np.random.rand(n) * eps
y = x * np.cos(r + 0.5 * x**3) + np.sin(0.5 * x**3)

v = np.vander(x, N = 20, increasing = True)
Q,R = np.linalg.qr(v)

QTb = Q.T @ y

def back_subst(A: np.ndarray, b: np.ndarray):
	n = b.shape[0]
	if A.shape[0] != A.shape[1]:
		raise ValueError("Input must be square, douche...")
	x = np.zeros(n)
	x[n-1] = b[n-1] / A[n-1, n-1]

	C = np.zeros((n,n))
	for i in range(n-2, -1, -1):
		x[i] =  (b[i] - A[i, i+1:] @ x[i+1:]) / A[i,i]
	return x





#AA = np.array([[1,2,3], [0,7,4], [0,0,3]])
#bb = np.ones(3)

coefs = back_subst(R, QTb)[::-1]

#print(coefs[::-1])
p = np.poly1d(coefs)
#print(p(xs).shape)


plt.plot(x, p(x), label = "lstsq", color = "r")
plt.plot(x,y, 'o')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()


'''exc 2'''

B = v.T @ v

def cholesky(A):
	n = A.shape[0]
	L = np.zeros_like(A, dtype = 'float')


	for i in range(n):
		s = sum(L[i, k]**2 for k in range(i))
		L[i, i] = np.sqrt(A[i,i] - s)

		for j in range(i+1, n):
			s = sum(L[j, k] * L[i,k] for k in range(i))
			L[j, i] = (A[j, i] - s) / L[i,i]

	return L


chol = cholesky(B)

L = cholesky(B)
print(np.allclose(B, L @ L.T, atol=1e-12, rtol=1e-12))
#print(L @ L.T)
#print(B)
