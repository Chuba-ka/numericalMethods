"""Задание 3.4, вариант 24: численное дифференцирование.

Запуск: python3 p_3_4.py
Используются округлённые табличные значения из условия.
"""

from math import isclose


def central_derivatives(xs, ys, x_star, points=3):
    """Первая и вторая производные на равномерной сетке.

    Три точки: дифференцирование квадратичного интерполянта, порядок O(h²).
    Пять точек: дифференцирование интерполянта степени 4, порядок O(h⁴).
    Указанные порядки относятся к погрешности формул для гладкой функции
    с точными значениями в узлах; округление таблицы в них не учитывается.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("Нужны не менее трёх узлов и столько же значений функции")
    if points not in (3, 5):
        raise ValueError("Допустимы формулы по трём или пяти точкам")
    h = xs[1] - xs[0]
    if h <= 0 or any(not isclose(xs[j + 1] - xs[j], h, rel_tol=1e-9, abs_tol=1e-12)
                     for j in range(len(xs) - 1)):
        raise ValueError("Требуется равномерная строго возрастающая сетка")
    i = next((j for j, x in enumerate(xs)
              if isclose(x, x_star, rel_tol=1e-12, abs_tol=1e-12)), None)
    if i is None:
        raise ValueError("Точка вычисления должна совпадать с узлом таблицы")
    radius = points // 2
    if i < radius or i + radius >= len(xs):
        raise ValueError("Недостаточно соседних узлов для центральной формулы")

    if points == 3:
        first = (ys[i + 1] - ys[i - 1]) / (2 * h)
        second = (ys[i + 1] - 2 * ys[i] + ys[i - 1]) / h**2
    else:
        first = (ys[i - 2] - 8 * ys[i - 1] + 8 * ys[i + 1] - ys[i + 2]) / (12 * h)
        second = (-ys[i - 2] + 16 * ys[i - 1] - 30 * ys[i]
                  + 16 * ys[i + 1] - ys[i + 2]) / (12 * h**2)
    return first, second


def main():
    xs = [1.0, 1.2, 1.4, 1.6, 1.8]
    ys = [2.0, 2.1344, 2.4702, 2.9506, 3.5486]
    x_star = 1.4

    print("Задание 3.4. Вариант 24. Численное дифференцирование")
    print(f"X* = {x_star}; h = {xs[1] - xs[0]:.1f}")
    print(f"\n{'i':>3} {'x_i':>8} {'y_i':>12}")
    for i, (x, y) in enumerate(zip(xs, ys)):
        print(f"{i:3d} {x:8.1f} {y:12.4f}")

    print("\nЦентральные формулы по трём точкам (i = 2):")
    print("f'(x_i) ≈ (y_{i+1} - y_{i-1}) / (2h)")
    print("f''(x_i) ≈ (y_{i+1} - 2y_i + y_{i-1}) / h²")
    first, second = central_derivatives(xs, ys, x_star)
    print(f"f'({x_star}) ≈ {first:.10f}")
    print(f"f''({x_star}) ≈ {second:.10f}")

    print("\nДополнительно: центральные формулы по пяти точкам:")
    print("f'(x_i) ≈ (y_{i-2} - 8y_{i-1} + 8y_{i+1} - y_{i+2}) / (12h)")
    print("f''(x_i) ≈ (-y_{i-2} + 16y_{i-1} - 30y_i + 16y_{i+1} - y_{i+2}) / (12h²)")
    first, second = central_derivatives(xs, ys, x_star, points=5)
    print(f"f'({x_star}) ≈ {first:.10f}")
    print(f"f''({x_star}) ≈ {second:.10f}")


if __name__ == '__main__':
    main()
