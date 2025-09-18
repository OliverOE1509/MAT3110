import numpy as np
import matplotlib.pyplot as plt





A = np.vander(x, N = 20, increasing = True)

chol = cholesky(A)
