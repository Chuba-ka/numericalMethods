import math



def mat_mult(A, B):
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def mat_transpose(A):
    n = len(A)
    m = len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def copy_mat(A):
    return [row[:] for row in A]


# ---------- QR-разложение методом отражений Хаусхолдера ----------

def qr_decomposition(A):
    """
    QR-разложение матрицы A (n x n) методом отражений Хаусхолдера.
    Возвращает Q (ортогональная) и R (верхнетреугольная), такие что A = Q * R.
    """
    n = len(A)
    R = copy_mat(A)
    Q = identity(n)

    for k in range(n - 1):
        # Столбец под диагональю (включая диагональный элемент)
        x = [R[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x < 1e-15:
            continue

        # Вектор Хаусхолдера
        alpha = -norm_x if x[0] >= 0 else norm_x
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(t * t for t in v))
        if norm_v < 1e-15:
            continue
        v = [t / norm_v for t in v]

        # Матрица отражения H_k (n x n), действующая на подматрицу [k:, k:]
        H = identity(n)
        for i in range(k, n):
            for j in range(k, n):
                H[i][j] -= 2 * v[i - k] * v[j - k]

        R = mat_mult(H, R)
        Q = mat_mult(Q, mat_transpose(H))

    return Q, R


# ---------- QR-алгоритм для нахождения собственных значений ----------

def subdiagonal_norm(A):
    """Норма поддиагональных элементов - критерий сходимости к (квази)треугольному виду."""
    n = len(A)
    s = 0.0
    for i in range(1, n):
        for j in range(i):
            s += A[i][j] ** 2
    return math.sqrt(s)


def eigenvalues_2x2_block(a, b, c, d):
    """Собственные значения блока [[a,b],[c,d]] (могут быть комплексными)."""
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    if disc >= 0:
        sq = math.sqrt(disc)
        return [(tr + sq) / 2, (tr - sq) / 2], False  # два вещественных, is_complex=False
    else:
        sq = math.sqrt(-disc)
        re = tr / 2
        im = sq / 2
        return [complex(re, im), complex(re, -im)], True


def qr_algorithm(A, eps, max_iter=1000):
    """
    QR-алгоритм для полной проблемы собственных значений произвольной
    (в общем случае несимметричной) матрицы.

    На каждой итерации: A_k = Q_k R_k,  A_{k+1} = R_k Q_k.
    При сходимости A_k стремится к верхней (квази)треугольной форме
    (форме Шура), на диагонали которой (либо в 2x2 блоках, если есть
    комплексно-сопряжённые собственные значения) стоят собственные значения.

    Возвращает:
        eigenvalues - список найденных собственных значений (вещественных и/или комплексных)
        Ak          - итоговая (квази)треугольная матрица
        history     - список (итерация, норма поддиагональных элементов)
        num_iter    - число выполненных итераций
    """
    n = len(A)
    Ak = copy_mat(A)
    history = [(0, subdiagonal_norm(Ak))]

    for it in range(1, max_iter + 1):
        Q, R = qr_decomposition(Ak)
        Ak = mat_mult(R, Q)
        norm = subdiagonal_norm(Ak)
        history.append((it, norm))
        if norm < eps:
            break

    num_iter = len(history) - 1

    # Извлечение собственных значений из квази-треугольной матрицы
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1:
            eigenvalues.append(Ak[i][i])
            i += 1
        else:
            below = Ak[i + 1][i]
            if abs(below) < 1e-6:
                eigenvalues.append(Ak[i][i])
                i += 1
            else:
                vals, is_complex = eigenvalues_2x2_block(
                    Ak[i][i], Ak[i][i + 1], Ak[i + 1][i], Ak[i + 1][i + 1]
                )
                eigenvalues.extend(vals)
                i += 2

    return eigenvalues, Ak, history, num_iter


def format_eig(v):
    if isinstance(v, complex):
        sign = "+" if v.imag >= 0 else "-"
        return f"{v.real:.6f} {sign} {abs(v.imag):.6f}i"
    return f"{v:.6f}"


if __name__ == "__main__":
    A = [
        [-3,  1, -1],
        [ 6,  9, -4],
        [ 5, -4, -8],
    ]

    eps = 1e-8
    eigenvalues, Ak, history, num_iter = qr_algorithm(A, eps)

    print("QR-разложение исходной матрицы (проверка):")
    Q, R = qr_decomposition(A)
    QR = mat_mult(Q, R)
    print("A =\n", "\n".join(str([f"{v:.4f}" for v in row]) for row in A))
    print("Q*R =\n", "\n".join(str([f"{v:.4f}" for v in row]) for row in QR))
    print()

    print(f"Число итераций QR-алгоритма до достижения точности eps={eps}: {num_iter}")
    print("\nИтоговая (квази)треугольная матрица A_k:")
    for row in Ak:
        print(["%.6f" % v for v in row])

    print("\nСобственные значения:")
    for i, lam in enumerate(eigenvalues, start=1):
        print(f"  lambda_{i} = {format_eig(lam)}")

    # Проверка через характеристический многочлен: сумма = след, произведение = det
    trace = sum(A[i][i] for i in range(len(A)))
    def det3(M):
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    det = det3(A)
    sum_eig = sum(eigenvalues)
    prod_eig = 1
    for e in eigenvalues:
        prod_eig *= e

    print("\nПроверка через инварианты матрицы:")
    print(f"  след(A) = {trace}, сумма собственных значений = {sum_eig}")
    print(f"  det(A) = {det}, произведение собственных значений = {prod_eig}")

    # Анализ зависимости погрешности от числа итераций
    print("\n=== Зависимость нормы поддиагональных элементов от числа итераций ===")
    print(f"{'Итерация':>10} | {'Норма поддиаг. эл-тов':>25}")
    for it, norm in history:
        print(f"{it:>10} | {norm:>25.10f}")

    print("\n=== Число итераций для достижения разной точности ===")
    print(f"{'eps':>12} | {'Итераций':>10}")
    for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]:
        _, _, _, ni = qr_algorithm(A, e)
        print(f"{e:>12.0e} | {ni:>10}")