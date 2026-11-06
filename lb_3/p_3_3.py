"""Задание 3.3, вариант 24: метод наименьших квадратов.

Запуск: python3 p_3_3.py
График сохраняется рядом со скриптом в p_3_3.svg, открываемый браузером.
Сторонние библиотеки не требуются.
"""

from pathlib import Path


def gaussian_solve(matrix, rhs):
    """Метод Гаусса с выбором главного элемента по столбцу."""
    n = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(augmented[i][k]))
        if augmented[pivot][k] == 0:
            raise ValueError("Система вырождена")
        augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
        for i in range(k + 1, n):
            factor = augmented[i][k] / augmented[k][k]
            augmented[i][k] = 0.0
            for j in range(k + 1, n + 1):
                augmented[i][j] -= factor * augmented[k][j]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = (augmented[i][n] - sum(
            augmented[i][j] * solution[j] for j in range(i + 1, n)
        )) / augmented[i][i]
    return solution


def least_squares(xs, ys, degree):
    """Решить нормальную систему Σ_j a_j Σ_i x_i^(k+j) = Σ_i y_i x_i^k.

    Коэффициенты возвращаются по возрастанию степеней.
    """
    if len(xs) != len(ys) or degree < 0 or len(set(xs)) < degree + 1:
        raise ValueError("Недостаточно различных узлов или некорректные данные")
    sums = [sum(x**power for x in xs) for power in range(2 * degree + 1)]
    matrix = [[sums[k + j] for j in range(degree + 1)] for k in range(degree + 1)]
    rhs = [sum(y * x**k for x, y in zip(xs, ys)) for k in range(degree + 1)]
    return gaussian_solve(matrix, rhs), matrix, rhs


def polynomial_value(coefficients, x):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    return result


def squared_error(xs, ys, coefficients):
    return sum((polynomial_value(coefficients, x) - y)**2 for x, y in zip(xs, ys))


def save_plot(xs, ys, polynomials, path):
    """Сохранить в SVG таблицу (точки с ломаной) и оба многочлена."""
    grid = [xs[0] + (xs[-1] - xs[0]) * i / 500 for i in range(501)]
    curves = [[polynomial_value(coefficients, x) for x in grid]
              for coefficients in polynomials]
    all_y = ys + [value for curve in curves for value in curve]
    span = max(all_y) - min(all_y)
    padding = 0.08 * span if span else 1.0
    ymin, ymax = min(all_y) - padding, max(all_y) + padding
    left, top, width, height = 85, 70, 760, 420

    def point(x, y):
        return (left + width * (x - xs[0]) / (xs[-1] - xs[0]),
                top + height * (ymax - y) / (ymax - ymin))

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="610" viewBox="0 0 920 610">',
           '<rect width="920" height="610" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#222">',
           '<text x="460" y="30" text-anchor="middle" font-size="20">МНК — вариант 24</text>']
    for k in range(6):
        y = ymin + (ymax - ymin) * k / 5
        _, py = point(xs[0], y)
        svg.append(f'<line x1="{left}" y1="{py}" x2="{left+width}" y2="{py}" stroke="#ddd"/>')
        svg.append(f'<text x="75" y="{py+5}" text-anchor="end">{y:.1f}</text>')
    for x in xs:
        px, _ = point(x, ymin)
        svg.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top+height}" stroke="#ddd"/>')
        svg.append(f'<text x="{px}" y="515" text-anchor="middle">{x:g}</text>')
    svg.extend([
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#333"/>',
        '<text x="870" y="515">x</text>', '<text x="65" y="60">y</text>',
    ])
    series = [(xs, ys, '#333', 'Табличная функция (ломаная)')]
    series += [(grid, curve, color, f'Многочлен степени {i}')
               for i, (curve, color) in enumerate(zip(curves, ['#1976d2', '#d84315']), 1)]
    for i, (x_values, y_values, color, label) in enumerate(series):
        points = ' '.join(f'{px:.3f},{py:.3f}' for px, py in
                          (point(x, y) for x, y in zip(x_values, y_values)))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        ly = 545 + 23 * i
        svg.append(f'<line x1="85" y1="{ly}" x2="120" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="130" y="{ly+5}">{label}</text>')
    for x, y in zip(xs, ys):
        px, py = point(x, y)
        svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="#333"/>')
    svg.append('</g></svg>')
    path.write_text('\n'.join(svg), encoding='utf-8')


def main():
    xs = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    ys = [100.0, 4.0, 1.2346, 0.59172, 0.34602, 0.22676]
    print("Задание 3.3. Вариант 24. Метод наименьших квадратов")
    print("Исходная таблица:")
    for x, y in zip(xs, ys):
        print(f"x = {x:.1f}; y = {y:.5f}")
    polynomials = []
    for degree in (1, 2):
        coefficients, matrix, rhs = least_squares(xs, ys, degree)
        polynomials.append(coefficients)
        print(f"\nСтепень {degree}. Нормальная система A*a = b:")
        for row, value in zip(matrix, rhs):
            print(' '.join(f'{element:14.8f}' for element in row) + f' | {value:14.8f}')
        formula = f"{coefficients[0]:.10f}"
        for power, coefficient in enumerate(coefficients[1:], 1):
            factor = '*x' if power == 1 else f'*x^{power}'
            formula += f" {'-' if coefficient < 0 else '+'} {abs(coefficient):.10f}{factor}"
        print(f"P_{degree}(x) = {formula}")
        print(f"Сумма квадратов ошибок: {squared_error(xs, ys, coefficients):.12f}")
        print(f"{'x_i':>8} {'y_i':>16} {'P(x_i)':>16} {'(P(x_i)-y_i)^2':>20}")
        for x, y in zip(xs, ys):
            value = polynomial_value(coefficients, x)
            print(f"{x:8.1f} {y:16.8f} {value:16.8f} {(value-y)**2:20.8f}")
    path = Path(__file__).resolve().with_suffix('.svg')
    save_plot(xs, ys, polynomials, path)
    print(f"\nГрафик сохранён: {path}")


if __name__ == '__main__':
    main()
