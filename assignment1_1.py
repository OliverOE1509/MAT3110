import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

n = 30
m=3
start = -2
stop = 2
x = np.linspace(start, stop, n)
eps = 1
np.random.seed(1)
r = np.random.rand(n) * eps
y = x * np.cos(r + 0.5 * x**3) + np.sin(0.5 * x**3)
y2 = 4 * x**5 - 5 * x**4 - 20 * x**3 + 10 * x**2 + 40 * x + 10 + r

v = np.vander(x, N = m, increasing = True)


def back_subst(A: np.ndarray, b: np.ndarray):
	n = b.shape[0]
	if A.shape[0] != A.shape[1]:
		raise ValueError("Input must be square, douche...")
	x = np.zeros(n)
	x[n-1] = b[n-1] / A[n-1, n-1]

	for i in range(n-2, -1, -1):
		x[i] =  (b[i] - A[i, i+1:] @ x[i+1:]) / A[i,i]
	return x


def cholesky(A):
	A = A.copy().astype(float)
	n = A.shape[0]
	L = np.zeros((n,n))
	D = np.zeros((n,n))
	for i in range(n):
		lk = A[:,i] / A[i,i]
		L[:,i] = lk
		D[i,i] = A[i,i] 
		A = A - D[i,i] * np.outer(lk, lk)
	return L, D


def cholesky2(B):
    n = B.shape[0]
    L = np.zeros_like(B)
    for i in range(n):
        L[i,i] = np.sqrt(B[i,i] - np.sum(L[i,:i]**2))
        for j in range(i+1, n):
            L[j,i] = (B[j,i] - np.sum(L[j,:i]*L[i,:i])) / L[i,i]
    return L

def forward_subst(A: np.ndarray, b: np.ndarray):
	'''A: nxn matrix, b: nx1 matrix'''
	n = A.shape[0]
	x = np.zeros(n)
	x[0] = b[0] / A[0,0]
	for i in range(1, n):
		x[i] = (b[i] - np.dot(A[i, :i], x[:i])) / A[i,i]
	return x

def solve_cholesky(V, y):
    B = V.T @ V
    L = np.linalg.cholesky(B)       # lower-triangular
    b = V.T @ y
    z = forward_subst(L, b)
    c = back_subst(L.T, z)
    return c


def fit_and_plot(dataset, m, x):
    V = np.vander(x, N=m, increasing=True)

    # --- QR method
    Q, R = np.linalg.qr(V)
    coefs_qr = back_subst(R, Q.T @ dataset)

    # --- Normal equations with Cholesky
    B = V.T @ V
    L = np.linalg.cholesky(B)
    b = V.T @ dataset
    z = forward_subst(L, b)
    coefs_chol = back_subst(L.T, z)

    # poly1d wants descending order
    p_qr = np.poly1d(coefs_qr[::-1])
    p_chol = np.poly1d(coefs_chol[::-1])

    # Plot
    xx = np.linspace(x.min(), x.max(), 400)
    plt.scatter(x, dataset, label="data")
    #plt.plot(xx, p_qr(xx), 'g-', label="QR fit")
    plt.plot(xx, p_chol(xx), 'r--', label="Cholesky fit")
    plt.legend()
    plt.show()


#fit_and_plot(dataset = y2, m=3, x = x)
fit_and_plot(dataset = y2, m=3, x = x)


