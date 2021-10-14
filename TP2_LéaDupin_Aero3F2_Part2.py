"""
------------------------------------------------
| DUPIN Léa | Aéro 3 F2                        |
| Ma313 : Algrébre linéaire numérique          |
| TP 2 - Partie 2 : Calcul des erreurs         |
------------------------------------------------
| IPSA Paris                                   |
| Année scolaire 2021 - 2022                   |
------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as pp
import time

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

def resoltrisup(T, b):

    n, m = T.shape
    x = np.zeros(n)
    for i in range (n - 1, - 1, - 1):
        S = T[i, i + 1:]@x[i + 1:]
        x[i] = (1 / T[i, i]) * (b[i] - S)

    return x

def ResolGS (A, b):

    Q = DecompositionGS(A)[0]
    R = DecompositionGS(A)[1]
    Y = np.dot(np.transpose(Q), b)
    T = resoltrisup(R, Y)

    return T

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

def ErreurSolve(n):

    A = Random_2A(n)
    b = Random_2b(n)
    x1 = ResolGS(A, b)
    x2 = np.linalg.solve(A, b)
    exp = np.linalg.norm(A * x1 - b)
    accepted = np.linalg.norm(A * x2 - b)
    erreur = (np.abs(accepted - exp) / accepted) * 100

    return erreur

N = range(10, 700, 50)
Erreur_calc = []

for k in N:
    Erreur_calc.append(ErreurSolve(k))

pp.plot(N, Erreur_calc, label = 'Erreur avec la résolution GS')
pp.xlabel('Taille de A')
pp.ylabel("Pourcentage d'erreur")
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

    return T

def ErreurSolve(n):

    A = Random_2A(n)
    b = Random_2b(n)
    x1 = ResolGS_np(A, b)
    x2 = np.linalg.solve(A, b)
    exp = np.linalg.norm(A * x1 - b)
    accepted = np.linalg.norm(A * x2 - b)
    erreur = (np.abs(accepted - exp) / accepted) * 100

    return erreur

N = range(10, 700, 50)
Erreur_calc = []

for k in N:
    Erreur_calc.append(ErreurSolve(k))

pp.plot(N, Erreur_calc, label = 'Erreur avec la résolution GS de numpy')
pp.xlabel('Taille de A')
pp.ylabel("Pourcentage d'erreur")
pp.legend()

pp.show()

# ----------------------------------------------

a = - 100
b = 100

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

def ErreurSolve(n):

    A = Random_2A(n)
    b = Random_2b(n)
    x1 = Gauss(A, b)
    x2 = np.linalg.solve(A, b)
    exp = np.linalg.norm(A * x1 - b)
    accepted = np.linalg.norm(A * x2 - b)
    erreur = (np.abs(accepted - exp) / accepted) * 100

    return erreur

N = range(10, 700, 50)
Erreur_calc = []

for k in N:
    Erreur_calc.append(ErreurSolve(k))

pp.plot(N, Erreur_calc, label = 'Erreur avec la décomposition LU de Gauss')
pp.xlabel('Taille de A')
pp.ylabel("Pourcentage d'erreur")
pp.legend()

pp.show()

# ----------------------------------------------

a = - 100
b = 100

def Cholesky(A, b):
    n, m = A.shape
    if n != m:
        print("A n'est pas carrée = problème")
        return

    L = np.zeros((n, n))

    Skk = 0
    Sik = 0
    
    for k in range(n):
        Skk = 0
        for j in range(0, k):
            Skk += (L[k, j]) ** 2
        L[k, k] = np.sqrt(np.abs(A[k, k] - Skk))
        for i in range(n): 
            Sik = 0
            if i > k:
                for j in range(0, k):
                    Sik += (L[i, j]) * (L[k, j])
                L[i, k] = (A[i, k] - Sik) / L[k, k]

    return L

def ErreurSolve(n):

    A = Random_2A(n)
    b = Random_2b(n)
    x1 = Cholesky(A, b)
    x2 = np.linalg.solve(A, b)
    exp = np.linalg.norm(A * x1 - b)
    accepted = np.linalg.norm(A * x2 - b)
    erreur = (np.abs(accepted - exp) / accepted) * 100

    return erreur

N = range(10, 500, 50)
Erreur_calc = []

for k in N:
    Erreur_calc.append(ErreurSolve(k))

pp.plot(N, Erreur_calc, label = 'Erreur avec la méthode de Cholesky')
pp.xlabel('Taille de A')
pp.ylabel("Pourcentage d'erreur")
pp.legend()

pp.show()
