import numpy as np




def LU(A):
	n = A.shape[0]
	L = np.eye(n, dtype = np.float64)
	U = A.copy()

	#print(U)

	for k in range(n):

		for j in range(k+1, n):
			print(U[k, k])
			factor = U[j, k] / U[k,k]
			#print(f'{U[j, k]} / {U[k,k]} = {factor}')
			L[j, k] = factor
			U[j, k:] -= factor * U[k, k:]

	return (L, U)


'''
Denne koden funker fordi for hver kolonne, så finner vi "factor" som er den j-te elementet i raden fra k, og deler den på pivot elementet
i pivot posisjonen (U[k,k]). Dermed sier vi at det j-te elementet i raden i den k-te kolonnen er lik "factor". I den første loopen, finner
koden at "factor" = 5/3. 
 '''


A = np.array([[3, 4, 5], [5, 6, 2], [1,2,4]], dtype = np.float64)

B = A * A.T
print(B)

exit()
L, U = LU(A)
#print(A)
print(L)
print(U)

