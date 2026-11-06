"""Задание 3.5, вариант 24: численное интегрирование.

Интеграл sqrt(16 - x²) на [-2, 2], шаги 1 и 0.5.
Запуск: python3 p_3_5.py. Сторонние библиотеки не требуются.
"""

from math import asin, fsum, isclose, isfinite, sqrt


def f(x):
    return sqrt(16 - x**2)


def interval_count(a, b, h):
    """Число отрезков; шаг должен делить длину интервала."""
    if not all(isfinite(value) for value in (a, b, h)) or h <= 0 or b <= a:
        raise ValueError("Нужны конечные a < b и положительный шаг")
    count = (b - a) / h
    n = round(count)
    if n < 1 or not isclose(count, n, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Длина интервала должна делиться на шаг без остатка")
    return n


def midpoint_rule(function, a, b, h):
    """Составная формула средних прямоугольников, порядок p = 2."""
    n = interval_count(a, b, h)
    return h * fsum(function(a + (i + 0.5) * h) for i in range(n))


def trapezoid_rule(function, a, b, h):
    """Составная формула трапеций, порядок p = 2."""
    n = interval_count(a, b, h)
    return h * ((function(a) + function(b)) / 2
                + fsum(function(a + i * h) for i in range(1, n)))


def simpson_rule(function, a, b, h):
    """Составная формула Симпсона, порядок p = 4."""
    n = interval_count(a, b, h)
    if n % 2:
        raise ValueError("Для формулы Симпсона число отрезков должно быть чётным")
    odd_sum = fsum(function(a + i * h) for i in range(1, n, 2))
    even_sum = fsum(function(a + i * h) for i in range(2, n, 2))
    return h / 3 * (function(a) + function(b) + 4 * odd_sum + 2 * even_sum)


def runge_romberg(coarse, fine, h_coarse, h_fine, order):
    """Вернуть уточнённый интеграл и оценку ошибки результата с мелким шагом.

    I_RR = I_fine + (I_fine - I_coarse) / ((h_coarse/h_fine)^p - 1).
    Оценка ошибки: |I_fine - I_coarse| / ((h_coarse/h_fine)^p - 1).
    """
    if not h_coarse > h_fine > 0 or order <= 0:
        raise ValueError("Требуются h_coarse > h_fine > 0 и положительный порядок")
    correction = (fine - coarse) / ((h_coarse / h_fine)**order - 1)
    return fine + correction, abs(correction)


def antiderivative(x):
    return x * sqrt(16 - x**2) / 2 + 8 * asin(x / 4)


def main():
    a, b = -2.0, 2.0
    h1, h2 = 1.0, 0.5
    exact = antiderivative(b) - antiderivative(a)
    methods = (
        ("Средние прямоугольники", midpoint_rule, 2),
        ("Трапеции", trapezoid_rule, 2),
        ("Симпсон", simpson_rule, 4),
    )
    print("Задание 3.5. Вариант 24")
    print("f(x) = sqrt(16 - x²), интервал [-2, 2]")
    print(f"h1 = {h1:g}; h2 = {h2:g}")
    print(f"Точное значение: 4*sqrt(3) + 8*pi/3 = {exact:.12f}")
    print("\nРунге — Ромберг: I_RR = I(h2) + (I(h2)-I(h1))/(2^p-1)")
    print("Оценка погрешности I(h2): R = |I(h2)-I(h1)|/(2^p-1)")
    for name, method, order in methods:
        coarse = method(f, a, b, h1)
        fine = method(f, a, b, h2)
        refined, estimate = runge_romberg(coarse, fine, h1, h2, order)
        print(f"\n{name}, p = {order}:")
        print(f"  I(h1) = {coarse:.12f}")
        print(f"  I(h2) = {fine:.12f}")
        print(f"  Оценка погрешности по Рунге — Ромбергу: {estimate:.12f}")
        print(f"  Уточнённое значение I_RR = {refined:.12f}")
        print(f"  Фактическая ошибка I(h1): {abs(coarse - exact):.12f}")
        print(f"  Фактическая ошибка I(h2): {abs(fine - exact):.12f}")
        print(f"  Фактическая ошибка I_RR:  {abs(refined - exact):.12f}")


if __name__ == '__main__':
    main()
