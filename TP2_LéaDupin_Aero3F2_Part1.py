"""
--------------------------------------------------
| DUPIN Léa | Aéro 3 F2                          |
| Ma313 : Algrébre linéaire numérique            |
| TP 2 - Partie 1 : Exercices et temps de calcul |
--------------------------------------------------
| IPSA Paris                                     |
| Année scolaire 2021 - 2022                     |
--------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as pp
import time

### Exercice 1 :

def DecompositionGS(A):
    
    n, m = A.shape
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    w = np.zeros(n)
   
    for j in range (0, n):
        for i in range (0, j):
                if i < j:
                    R[i,j] = np.dot(A[:,j], Q[:,i])
                else:
                    pass

        S = 0

        for k in range (0, j):
            S = S + (R[k, j] * Q[:, k])
        
        w = A[:, j] - S
        R[j, j] = np.linalg.norm(w)
        Q[:, j] = (1 / R[j, j]) * w

    return Q, R, S

A = np.array([[6, 6, 16], [-3, -9, -2], [6, -6, -8]])
print("Décomposition GS : ", DecompositionGS(A))


### Exercice 2 :

def resoltrisup(T, b):

    n, m = T.shape
    x = np.zeros(n)
  
    for i in range (n - 1, - 1, - 1):
        S = T[i, i + 1:]@x[i + 1:]
        x[i] = (1 / T[i, i]) * (b[i] - S)

    return x

def ResolGS(A, b):

    Q = DecompositionGS(A)[0]
    R = DecompositionGS(A)[1]
    Y = np.dot(np.transpose(Q), b)
    T = resoltrisup(R, Y)
    print("Résolution triangulaire supérieure : ", T)

    return(T)


### Exercice 3 :

def Gauss(A, b):

    A = A.copy()
    b = b.copy()
    n = b.size
    
    for i in range(n):
        for j in range(i + 1, n):
            g = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - g * A[i, :]
            b[j] = b[j] - g * b[i]

    x = resoltrisup(A, b)

    return x

a = - 100
b = 100

def RandomA(n):

    X = np.random.rand(n, n)
    A_n = (b - a) * X + a

    return A_n

def Randomb(n):

    Y = np.random.rand(n, 1)
    b_n = (b - a) * Y + a

    return b_n


def TpsResolQR(n):

    start = time.process_time()
    ResolGS(RandomA(n), Randomb(n))
    end = time.process_time()
    t = end - start

    return t

def TpsResolGauss(n):

    start = time.process_time()
    Gauss(RandomA(n), Randomb(n))
    end = time.process_time()
    t = end - start

    return t

N = range(10, 700, 50)
LTQR = []
LTGauss = []

for k in N:
    LTQR.append(TpsResolQR(k))
    LTGauss.append(TpsResolGauss(k))

pp.plot(N, LTQR, label = 'Décomposition QR')
pp.plot(N, LTGauss, label = 'Décomposition LU de Gauss')
pp.xlabel('Taille de A')
pp.ylabel('Temps de calcul')
pp.legend()

pp.show()

# ----------------------------------------------

a = - 100
b = 100

def Cholesky(A, b):
    n, m = A.shape
    if n != m:
        print ("A n'est pas carrée = problème")
        return

    L = np.zeros((n, n))

    Skk = 0
    Sik = 0
    
    for k in range(n):
        Skk = 0
        for j in range(0, k):
            Skk += (L[k, j]) ** 2
            
        L[k, k] = np.sqrt(A[k, k] - Skk)

        for i in range(n):
            Sik = 0
            if i > k:
                for j in range(0, k):
                    Sik += (L[i, j]) * (L[k, j])
                L[i, k] = (A[i, k] - Sik) / L[k, k]

    return L

a = - 100
b = 100

def Random_2A(n):

    X = np.random.rand(n, n)
    A_n = (b - a) * X + a

    return A_n

def Random_2b(n):

    Y = np.random.rand(n, 1)
    b_n = (b - a) * Y + a

    return b_n


def TpsResolQR(n):

    start = time.process_time()
    ResolGS(Random_2A(n), Random_2b(n))
    end = time.process_time()
    t = end - start

    return t

def TpsResolCholesky(n):

    start = time.process_time()
    Cholesky(Random_2A(n), Random_2b(n))
    end = time.process_time()
    t = end - start

    return t

N = range(10, 700, 50)
LTQR = []
LTCholesky = []

for k in N:
    LTQR.append(TpsResolQR(k))
    LTCholesky.append(TpsResolCholesky(k))

pp.plot(N, LTQR, label = 'Décomposition QR')
pp.plot(N, LTCholesky, label = 'Méthode de Cholesky')
pp.xlabel('Taille de A')
pp.ylabel('Temps de calcul')
pp.legend()

pp.show()

# ----------------------------------------------

a = - 100
b = 100

def ResolGS_np(A, b):

    Q = np.linalg.qr(A)[0]
    R = np.linalg.qr(A)[1]
    Y = np.dot(np.transpose(Q), b)
    T = resoltrisup(R, Y)
    print("Résolution triangulaire supérieure : ", T)

    return(T)

def Random_2A(n):

    X = np.random.rand(n, n)
    A_n = (b - a) * X + a

    return A_n

def Random_2b(n):

    Y = np.random.rand(n, 1)
    b_n = (b - a) * Y + a

    return b_n

def TpsResolQR(n):

    start = time.process_time()
    ResolGS(Random_2A(n), Random_2b(n))
    end = time.process_time()
    t = end - start

    return t

def TpsResolQR_np(n):

    start = time.process_time()
    ResolGS_np(Random_2A(n), Random_2b(n))
    end = time.process_time()
    t = end - start

    return t

N = range(10, 700, 50)
LTQR = []
LTQR_np = []

for k in N:
    LTQR.append(TpsResolQR(k))
    LTQR_np.append(TpsResolQR_np(k))

pp.plot(N, LTQR, label = 'Décomposition QR')
pp.plot(N, LTQR_np, label = 'Décomposition QR numpy')
pp.xlabel('Taille de A')
pp.ylabel('Temps de calcul')
pp.legend()

pp.show()

# ----------------------------------------------

def Random_2A(n):

    X = np.random.rand(n, n)
    A_n = (b - a) * X + a

    return A_n

def Random_2b(n):

    Y = np.random.rand(n, 1)
    b_n = (b - a) * Y + a

    return b_n

def TpsResolQR(n):

    start = time.process_time()
    ResolGS(Random_2A(n), Random_2b(n))
    end = time.process_time()
    t = end - start

    return t

def TpsResolSolve(n):

    start = time.process_time()
    np.linalg.solve(Random_2A(n), Random_2b(n))
    end = time.process_time()
    t = end - start

    return t

N = range(10, 700, 50)
LTQR = []
LTSolve = []

for k in N:
    LTQR.append(TpsResolQR(k))
    LTSolve.append(TpsResolSolve(k))

pp.plot(N, LTQR, label = 'Décomposition QR')
pp.plot(N, LTSolve, label = 'Numpy solveur')
pp.xlabel('Taille de A')
pp.ylabel('Temps de calcul')
pp.legend()

pp.show()
