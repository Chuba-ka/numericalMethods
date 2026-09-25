import math


def off_diagonal_norm(A):
    """Норма (сумма квадратов) внедиагональных элементов симметричной матрицы."""
    n = len(A)
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += A[i][j] ** 2
    return math.sqrt(s)


def max_off_diagonal(A):
    """Находит индексы (p, q) наибольшего по модулю внедиагонального элемента."""
    n = len(A)
    p, q = 0, 1
    max_val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
    return p, q, max_val


def jacobi_eigen(A, eps, max_iter=1000):
    """
    Метод вращений Якоби для нахождения собственных значений и собственных
    векторов симметричной матрицы A.

    Параметры:
        A        - симметричная матрица (список списков)
        eps      - требуемая точность (норма внедиагональных элементов)
        max_iter - максимальное число итераций

    Возвращает:
        eigenvalues  - список собственных значений (диагональ итоговой матрицы)
        eigenvectors - матрица, столбцы которой - собственные векторы
        history      - список пар (номер итерации, норма внедиагональных элементов)
        num_iter     - фактическое число выполненных итераций
    """
    n = len(A)
    Ak = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    history = [(0, off_diagonal_norm(Ak))]

    for it in range(1, max_iter + 1):
        p, q, max_val = max_off_diagonal(Ak)

        if abs(Ak[p][q]) < 1e-15:
            break

        # Угол поворота
        if Ak[p][p] == Ak[q][q]:
            theta = math.pi / 4 if Ak[p][q] > 0 else -math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * Ak[p][q], Ak[p][p] - Ak[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # Матрица вращения применяется к Ak: Ak_new = R^T * Ak * R
        A_new = [row[:] for row in Ak]

        for i in range(n):
            if i != p and i != q:
                A_new[i][p] = c * Ak[i][p] + s * Ak[i][q]
                A_new[p][i] = A_new[i][p]
                A_new[i][q] = -s * Ak[i][p] + c * Ak[i][q]
                A_new[q][i] = A_new[i][q]

        A_new[p][p] = c * c * Ak[p][p] + 2 * s * c * Ak[p][q] + s * s * Ak[q][q]
        A_new[q][q] = s * s * Ak[p][p] - 2 * s * c * Ak[p][q] + c * c * Ak[q][q]
        A_new[p][q] = 0.0
        A_new[q][p] = 0.0

        Ak = A_new

        # Накопление собственных векторов: V = V * R
        for i in range(n):
            v_ip = V[i][p]
            v_iq = V[i][q]
            V[i][p] = c * v_ip + s * v_iq
            V[i][q] = -s * v_ip + c * v_iq

        norm = off_diagonal_norm(Ak)
        history.append((it, norm))

        if norm < eps:
            break

    eigenvalues = [Ak[i][i] for i in range(n)]
    eigenvectors = V  # столбец j - собственный вектор для eigenvalues[j]

    return eigenvalues, eigenvectors, history, len(history) - 1


def print_vector(v):
    return "[" + ", ".join(f"{c:.6f}" for c in v) + "]"


def verify(A, eigenvalues, eigenvectors):
    """Проверка A*v = lambda*v для каждой найденной пары."""
    n = len(A)
    print("Проверка (A*v сравнивается с lambda*v):")
    for k in range(n):
        v = [eigenvectors[i][k] for i in range(n)]
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        lv = [eigenvalues[k] * v[i] for i in range(n)]
        err = max(abs(Av[i] - lv[i]) for i in range(n))
        print(f"  lambda_{k+1} = {eigenvalues[k]:.6f}: "
              f"A*v = {print_vector(Av)}, lambda*v = {print_vector(lv)}, "
              f"расхождение = {err:.2e}")


if __name__ == "__main__":
    A = [
        [-8, -4,  8],
        [-4, -3,  9],
        [ 8,  9, -5],
    ]

    eps = 1e-6
    eigenvalues, eigenvectors, history, num_iter = jacobi_eigen(A, eps)

    print("Собственные значения:")
    for i, lam in enumerate(eigenvalues, start=1):
        print(f"  lambda_{i} = {lam:.8f}")

    print("\nСобственные векторы (столбцы):")
    n = len(A)
    for k in range(n):
        v = [eigenvectors[i][k] for i in range(n)]
        print(f"  v_{k+1} = {print_vector(v)}")

    print(f"\nЧисло итераций до достижения точности eps={eps}: {num_iter}")
    print()
    verify(A, eigenvalues, eigenvectors)

    # Анализ зависимости погрешности (нормы внедиагональных элементов) от числа итераций
    print("\n=== Зависимость нормы внедиагональных элементов от числа итераций ===")
    print(f"{'Итерация':>10} | {'Норма внедиаг. эл-тов':>25}")
    for it, norm in history:
        print(f"{it:>10} | {norm:>25.10f}")

    # Число итераций для разных требуемых точностей
    print("\n=== Число итераций для достижения разной точности ===")
    print(f"{'eps':>12} | {'Итераций':>10}")
    for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        _, _, _, ni = jacobi_eigen(A, e)
        print(f"{e:>12.0e} | {ni:>10}")