import numpy as np
from numpy.linalg import norm

def lu_decomposition_with_pivoting(A):
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    pivots = 0

    for k in range(n - 1):
        pivot_row = np.argmax(np.abs(U[k:, k])) + k
        
        if abs(U[pivot_row, k]) < 1e-14:
            raise ValueError("Матрица вырождена (нулевой главный элемент)")

        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
            pivots += 1

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U, pivots


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


def solve_lu(P, L, U, b):
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = back_substitution(U, y)
    return x


def determinant_from_lu(U, pivots):
    det = np.prod(np.diag(U))
    if pivots % 2 == 1:
        det = -det
    return det


def inverse_from_lu(P, L, U):
    n = P.shape[0]
    inv = np.zeros((n, n))
    I = np.eye(n)
    for i in range(n):
        inv[:, i] = solve_lu(P, L, U, I[:, i])
    return inv


A = np.array([
    [-7, -2, -1, -4],
    [-4,  6,  0, -4],
    [-8,  2, -9, -3],
    [ 0,  0, -7,  1]
], dtype=float)

b = np.array([-12, 22, 51, 49], dtype=float)

print("Матрица A:")
print(A)
print("\nВектор b:")
print(b)

P, L, U, pivots = lu_decomposition_with_pivoting(A)

print("\nМатрица перестановок P:")
print(P)
print("\nНижняя треугольная L:")
print(L)
print("\nВерхняя треугольная U:")
print(U)

print("\nПроверка PA - LU (норма разности):")
print(norm(P @ A - L @ U))

x = solve_lu(P, L, U, b)
print("\nРешение системы x =")
print(x)

print("\nПроверка Ax - b (норма невязки):")
print(norm(A @ x - b))

det = determinant_from_lu(U, pivots)
print("\nОпределитель det(A) =", det)

A_inv = inverse_from_lu(P, L, U)
print("\nОбратная матрица A^{-1}:")
print(A_inv)

print("\nПроверка A * A^{-1} (норма отклонения от I):")
print(norm(A @ A_inv - np.eye(4)))