import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def back_subst(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = U[i, i+1:] @ x[i+1:]
        x[i] = (b[i] - s) / U[i, i]
    return x

def forward_subst(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        s = L[i, :i] @ x[:i]
        x[i] = (b[i] - s) / L[i, i]
    return x

def fit_and_plot(x, y, m, title=""):
    # Vandermonde
    V = np.vander(x, N=m, increasing=True)
    
    # --- QR method ---
    Q, R = np.linalg.qr(V)
    coefs_qr = back_subst(R, Q.T @ y)
    
    # --- Normal equations (Cholesky) ---
    B = V.T @ V
    Rchol = np.linalg.cholesky(B)         # upper triangular
    z = forward_subst(Rchol.T, V.T @ y)   # solve R^T z = V^T y
    coefs_chol = back_subst(Rchol, z)     # solve R c = z
    
    # Poly1d expects descending order
    p_qr   = np.poly1d(coefs_qr[::-1])
    p_chol = np.poly1d(coefs_chol[::-1])
    
    # Plot
    xx = np.linspace(x.min(), x.max(), 400)
    plt.figure()
    plt.plot(x, y, 'o', label="data")
    plt.plot(xx, p_qr(xx), '-', label="QR fit")
    plt.plot(xx, p_chol(xx), '--', label="Cholesky fit")
    plt.title(title)
    plt.legend()
    plt.show()
    
    # optional: check difference
    print(f"{title}: max |QR - Cholesky| coeff difference =",
          np.max(np.abs(coefs_qr - coefs_chol)))

# -----------------------------
# Example data
# -----------------------------
n = 30
x = np.linspace(-2, 2, n)
eps = 1
np.random.seed(1)
r = np.random.rand(n) * eps
y  = x * np.cos(r + 0.5 * x**3) + np.sin(0.5 * x**3)
y2 = x * (np.cos(r + 0.5 * x) + np.sin(0.5 * x**3))

# Degree m=3 (for example)
fit_and_plot(x, y2, 3, title="Dataset 1")
#fit_and_plot(x, y2, 3, title="Dataset 2")
