"""Задание 3.1, вариант 24: интерполяция функции f(x) = 1 / x².

Запуск: python3 p_3_1.py
Коэффициенты многочленов хранятся по возрастанию степеней x.
"""


def f(x):
    return 1.0 / x**2


def multiply_by_linear(coefficients, root):
    """Умножить многочлен на (x - root)."""
    result = [0.0] * (len(coefficients) + 1)
    for i, coefficient in enumerate(coefficients):
        result[i] -= root * coefficient
        result[i + 1] += coefficient
    return result


def lagrange_polynomial(xs, ys):
    """L(x) = Σ y_i ∏_{j≠i} (x - x_j) / (x_i - x_j)."""
    result = [0.0] * len(xs)
    for i in range(len(xs)):
        basis = [1.0]
        denominator = 1.0
        for j in range(len(xs)):
            if j != i:
                basis = multiply_by_linear(basis, xs[j])
                denominator *= xs[i] - xs[j]
        for k, coefficient in enumerate(basis):
            result[k] += ys[i] * coefficient / denominator
    return result


def divided_differences(xs, ys):
    """Таблица разделённых разностей: table[i][k] = f[x_i, ..., x_{i+k}]."""
    n = len(xs)
    table = [[y] + [0.0] * (n - 1) for y in ys]
    for order in range(1, n):
        for i in range(n - order):
            table[i][order] = (
                (table[i + 1][order - 1] - table[i][order - 1])
                / (xs[i + order] - xs[i])
            )
    return table


def newton_value(xs, coefficients, x):
    """Вычислить многочлен Ньютона вложенным умножением."""
    result = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        result = coefficients[i] + (x - xs[i]) * result
    return result


def lagrange_value(xs, ys, x):
    """Вычислить значение непосредственно по формуле Лагранжа."""
    result = 0.0
    for i, y in enumerate(ys):
        term = y
        for j in range(len(xs)):
            if j != i:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        result += term
    return result


def polynomial_text(coefficients):
    terms = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        factor = "" if power == 0 else ("*x" if power == 1 else f"*x^{power}")
        term = f"{abs(coefficient):.10f}{factor}"
        if not terms:
            terms.append(("-" if coefficient < 0 else "") + term)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(terms)


def solve(label, xs, x_star):
    ys = [f(x) for x in xs]
    table = divided_differences(xs, ys)
    newton_coefficients = table[0]
    print(f"\nПункт {label}: узлы {xs}")
    print("Таблица значений и разделённых разностей:")
    print(f"{'x_i':>8} {'y_i':>16} {'1-й порядок':>16} {'2-й порядок':>16} {'3-й порядок':>16}")
    for i, x in enumerate(xs):
        values = " ".join(f"{value:16.10f}" for value in table[i][:len(xs) - i])
        print(f"{x:8.1f} {values}")

    print("\nМногочлен Лагранжа (раскрытые скобки):")
    print("L_3(x) = " + polynomial_text(lagrange_polynomial(xs, ys)))
    newton_text = f"{newton_coefficients[0]:.10f}"
    for i in range(1, len(xs)):
        coefficient = newton_coefficients[i]
        factors = "".join(f"*(x - {root:g})" for root in xs[:i])
        newton_text += f" {'-' if coefficient < 0 else '+'} {abs(coefficient):.10f}{factors}"
    print("Многочлен Ньютона:")
    print("N_3(x) = " + newton_text)

    exact = f(x_star)
    print(f"\nX* = {x_star:g}; f(X*) = {exact:.12f}")
    for name, value in (
        ("Лагранж", lagrange_value(xs, ys, x_star)),
        ("Ньютон", newton_value(xs, newton_coefficients, x_star)),
    ):
        print(f"{name}: P(X*) = {value:.12f}; |f(X*) - P(X*)| = {abs(exact - value):.12f}")


def main():
    print("Задание 3.1. Вариант 24. f(x) = 1/x²")
    solve("а", [0.1, 0.5, 0.9, 1.3], 0.8)
    solve("б", [0.1, 0.5, 1.1, 1.3], 0.8)


if __name__ == "__main__":
    main()
