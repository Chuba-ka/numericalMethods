"""ЛР 2.2: система 3*x1**2-x1+x2**2-1=0, x2-tan(x1)=0."""
import argparse
import csv
import math
from decimal import Decimal, localcontext
from pathlib import Path
from method_1 import svg_plot

A = 3
X0 = (0.6, 0.7)


def g(y):
    return (1 + math.sqrt(13 - 12*y*y))/6


Q = (2*0.75/math.sqrt(13-12*0.75**2)) / math.cos(g(0.65))**2


def f(x):
    u, v = x
    return A*u*u-u+v*v-1, v-math.tan(u)


def jacobian(x):
    u, v = x
    return ((2*A*u-1, 2*v), (-1/math.cos(u)**2, 1))


def phi(x):
    u = g(x[1])
    return u, math.tan(u)


def norm(x):
    return max(abs(v) for v in x)


def distance(x, y):
    return norm([a-b for a,b in zip(x,y)])


def in_domain(x):
    return 0.58 <= x[0] <= 0.65 and 0.65 <= x[1] <= 0.75


def newton_step(x):
    (a,b),(c,d) = jacobian(x)
    r,s = f(x)
    det = a*d-b*c
    if abs(det) < 1e-15:
        raise RuntimeError('Матрица Якоби вырождена.')
    # Решение J(x)*delta = -F(x) для матрицы 2x2.
    return x[0]+(-d*r+b*s)/det, x[1]+(c*r-a*s)/det


def error_bound(x):
    return distance(x, phi(x))/(1-Q)


def solve(eps, method, max_iter=1000):
    if not math.isfinite(eps) or eps < 1e-14:
        raise ValueError('Точность должна быть конечной и не меньше 1e-14.')
    if method not in ('iteration', 'newton'):
        raise ValueError('Неизвестный метод.')
    x = X0
    history = [(0, *x, norm(f(x)), error_bound(x))]
    for k in range(1, max_iter+1):
        y = phi(x) if method == 'iteration' else newton_step(x)
        if not in_domain(y):
            raise RuntimeError('Приближение вышло из области обоснованной сходимости.')
        bound = Q/(1-Q)*distance(y,x) if method == 'iteration' else error_bound(y)
        history.append((k, *y, norm(f(y)), bound))
        if bound <= eps:
            return y, history
        x = y
    raise RuntimeError('Превышено максимальное число итераций.')


def decimal_tan(x):
    """sin/cos через ряды Тейлора для |x| < 1, с Decimal."""
    sin_term, cos_term = x, Decimal(1)
    sin_sum, cos_sum = sin_term, cos_term
    for n in range(1, 65):
        sin_term *= -x*x/Decimal((2*n)*(2*n+1))
        cos_term *= -x*x/Decimal((2*n-1)*(2*n))
        sin_sum += sin_term
        cos_sum += cos_term
    return sin_sum/cos_sum


def reference_root():
    with localcontext() as ctx:
        ctx.prec = 70
        lo, hi = Decimal('0.58'), Decimal('0.65')
        for _ in range(200):
            u = (lo+hi)/2
            v = decimal_tan(u)
            if 3*u*u-u+v*v-1 > 0:
                hi = u
            else:
                lo = u
        u = (lo+hi)/2
        return +u, +decimal_tan(u)


def actual_error(x, reference):
    with localcontext() as ctx:
        ctx.prec = 70
        return float(max(abs(Decimal.from_float(v)-r) for v,r in zip(x,reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--output', type=Path, default=Path('results_2'))
    args = parser.parse_args()
    try:
        eps = args.eps if args.eps is not None else float(input('Введите точность eps (например 1e-8): '))
        results = [(name, *solve(eps, method)) for name,method in [('Простая итерация','iteration'), ('Ньютон','newton')]]
    except ValueError as exc:
        parser.error(str(exc))
    reference = reference_root()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'a = {A}; x0 = {X0}; q <= {Q:.8f}')
    print(f'Эталон: x1 = {reference[0]:.25f}, x2 = {reference[1]:.25f}')
    curves = []
    for name, root, history in results:
        print(f'\n{name}: x1 = {root[0]:.15f}, x2 = {root[1]:.15f}; итераций: {len(history)-1}')
        print(' k       x1                 x2            ||F||∞       оценка       ошибка')
        rows = [(k,u,v,res,bound,actual_error((u,v), reference)) for k,u,v,res,bound in history]
        for k,u,v,res,bound,err in rows:
            print(f'{k:2d} {u:.15f} {v:.15f} {res:.3e} {bound:.3e} {err:.3e}')
        filename = 'iteration.csv' if name == 'Простая итерация' else 'newton.csv'
        with (args.output/filename).open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['k','x1','x2','residual_inf','error_bound','actual_error_inf'])
            writer.writerows(rows)
        curves.append((name, [(row[0],row[-1]) for row in rows]))
    svg_plot(args.output/'errors.svg', curves, 'Количество итераций k', 'Погрешность в норме max', 'Погрешность решения системы', log=True)
    points = [i/1000 for i in range(768)]
    ellipse = [(u, math.sqrt(max(0,1+u-3*u*u))) for u in points if 1+u-3*u*u >= 0]
    svg_plot(args.output/'system.svg', [('x₂ = √(1 + x₁ − 3x₁²)', ellipse), ('x₂ = tan(x₁)', [(u,math.tan(u)) for u in points]), ('Начальное приближение', [X0, (X0[0]+0.0001, X0[1]+0.0001)])], 'x₁', 'x₂', 'Графическое определение положительного решения')
    print(f'\nТаблицы и графики: {args.output.resolve()}')


if __name__ == '__main__':
    main()
