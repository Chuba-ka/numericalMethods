import copy


def check_diagonal_dominance(A):
    """Проверка достаточного условия сходимости - диагонального преобладания.
    """
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_diag_sum:
            return False
    return True


def jacobi_method(A, b, eps, max_iter=1000, x0=None):
    """
    Метод простых итераций (метод Якоби) для решения СЛАУ Ax = b.

    Параметры:
        A       - матрица системы (список списков)
        b       - вектор правых частей
        eps     - требуемая точность
        max_iter - максимальное число итераций
        x0      - начальное приближение (по умолчанию нулевой вектор)

    Возвращает:
        x            - найденное решение
        iterations   - список приближений на каждой итерации
        num_iter     - количество выполненных итераций
    """
    n = len(A)
    x = x0[:] if x0 is not None else [0.0] * n
    iterations = [x[:]]

    for it in range(1, max_iter + 1):
        x_new = [0.0] * n
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i][j] * x[j]
            x_new[i] = s / A[i][i]

        # критерий остановки: норма разности последовательных приближений
        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        iterations.append(x_new[:])
        x = x_new

        if diff < eps:
            return x, iterations, it

    return x, iterations, max_iter


def seidel_method(A, b, eps, max_iter=1000, x0=None):
    """
    Метод Зейделя (Гаусса-Зейделя) для решения СЛАУ Ax = b.
    В отличие от метода Якоби, использует уже обновлённые компоненты x
    в пределах текущей итерации.
    """
    n = len(A)
    x = x0[:] if x0 is not None else [0.0] * n
    iterations = [x[:]]

    for it in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i][j] * x[j]   # x[j] уже обновлено для j < i
            x[i] = s / A[i][i]

        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        iterations.append(x[:])

        if diff < eps:
            return x, iterations, it

    return x, iterations, max_iter


def print_report(name, x, num_iter, A, b):
    print(f"--- {name} ---")
    for i, xi in enumerate(x, start=1):
        print(f"x{i} = {xi:.8f}")
    print(f"Число итераций до достижения точности: {num_iter}")
    print("Проверка (A*x сравнивается с b):")
    n = len(b)
    for i in range(n):
        lhs = sum(A[i][j] * x[j] for j in range(n))
        print(f"  Уравнение {i+1}: получено {lhs:.6f}, требуется {b[i]}")
    print()


if __name__ == "__main__":
    A = [
        [-25,  4, -4,  9],
        [ -9, 21,  5, -6],
        [  9,  2, 19, -7],
        [ -7,  4, -7, 25],
    ]
    b = [86, 29, 28, 68]

    print("Матрица диагонально преобладающая:", check_diagonal_dominance(A))
    print()

    eps = 1e-6

    x_j, hist_j, n_j = jacobi_method(A, b, eps)
    print_report("Метод простых итераций (Якоби)", x_j, n_j, A, b)

    x_s, hist_s, n_s = seidel_method(A, b, eps)
    print_report("Метод Зейделя", x_s, n_s, A, b)

    # Анализ числа итераций при разной точности
    print("=== Анализ числа итераций при разной точности ===")
    print(f"{'eps':>12} | {'Якоби':>10} | {'Зейдель':>10}")
    for e in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        _, _, nj = jacobi_method(A, b, e)
        _, _, ns = seidel_method(A, b, e)
        print(f"{e:>12.0e} | {nj:>10} | {ns:>10}")