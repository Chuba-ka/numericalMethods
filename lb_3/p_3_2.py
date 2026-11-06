"""Задание 3.2, вариант 24: естественный кубический сплайн.

Запуск: python3 p_3_2.py
Используются табличные значения из PDF без пересчёта функции.
"""

from bisect import bisect_right


def thomas_method(lower, diagonal, upper, rhs):
    """Метод прогонки для трёхдиагональной системы.

    Все массивы имеют длину n; lower[0] = upper[n - 1] = 0.
    """
    n = len(diagonal)
    p = [0.0] * n
    q = [0.0] * n
    for i in range(n):
        denominator = diagonal[i] + (lower[i] * p[i - 1] if i else 0.0)
        if denominator == 0:
            raise ValueError("Нулевой знаменатель в методе прогонки")
        p[i] = -upper[i] / denominator
        q[i] = (rhs[i] - (lower[i] * q[i - 1] if i else 0.0)) / denominator
    solution = [0.0] * n
    solution[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = p[i] * solution[i + 1] + q[i]
    return solution


def natural_cubic_spline(xs, ys):
    """Вернуть коэффициенты (a, b, c, d) каждого интервала.

    S_i(x) = a_i + b_i*t + c_i*t² + d_i*t³, t = x - x_i.
    Сначала вычисляются M_i = S''(x_i), причём M_0 = M_n = 0.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Нужны не менее двух узлов и столько же значений функции")
    h = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    if any(step <= 0 for step in h):
        raise ValueError("Узлы должны строго возрастать")
    n = len(xs)
    lower = [0.0] * n
    diagonal = [1.0] * n
    upper = [0.0] * n
    rhs = [0.0] * n
    for i in range(1, n - 1):
        # h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1}
        # = 6 ((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1}).
        lower[i] = h[i - 1]
        diagonal[i] = 2 * (h[i - 1] + h[i])
        upper[i] = h[i]
        rhs[i] = 6 * ((ys[i + 1] - ys[i]) / h[i]
                      - (ys[i] - ys[i - 1]) / h[i - 1])
    moments = thomas_method(lower, diagonal, upper, rhs)
    coefficients = []
    for i, step in enumerate(h):
        a = ys[i]
        b = (ys[i + 1] - ys[i]) / step - step * (2 * moments[i] + moments[i + 1]) / 6
        c = moments[i] / 2
        d = (moments[i + 1] - moments[i]) / (6 * step)
        coefficients.append((a, b, c, d))
    return coefficients


def spline_value(xs, coefficients, x):
    """Вычислить сплайн на отрезке интерполяции."""
    if not xs[0] <= x <= xs[-1]:
        raise ValueError("Точка находится вне отрезка интерполяции")
    i = min(bisect_right(xs, x) - 1, len(coefficients) - 1)
    a, b, c, d = coefficients[i]
    t = x - xs[i]
    return a + t * (b + t * (c + t * d))


def main():
    xs = [0.1, 0.5, 0.9, 1.3, 1.7]
    ys = [100.00, 4.0, 1.2346, 0.59172, 0.34602]
    x_star = 0.8
    coefficients = natural_cubic_spline(xs, ys)

    print("Задание 3.2. Вариант 24. Естественный кубический сплайн")
    print("Условия на концах: S''(0.1) = S''(1.7) = 0")
    print("\nИсходная таблица:")
    print(f"{'i':>3} {'x_i':>8} {'f_i':>12}")
    for i, (x, y) in enumerate(zip(xs, ys)):
        print(f"{i:3d} {x:8.1f} {y:12.5f}")

    print("\nS_i(x) = a_i + b_i*t + c_i*t² + d_i*t³, t = x - x_i")
    print(f"{'Интервал':>12} {'a_i':>16} {'b_i':>16} {'c_i':>16} {'d_i':>16}")
    for i, row in enumerate(coefficients):
        interval = f"[{xs[i]:.1f}, {xs[i + 1]:.1f}]"
        print(f"{interval:>12} " + " ".join(f"{value:16.10f}" for value in row))
    print(f"\nS({x_star:g}) = {spline_value(xs, coefficients, x_star):.12f}")


if __name__ == "__main__":
    main()
