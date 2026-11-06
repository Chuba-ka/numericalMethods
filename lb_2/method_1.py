"""Решение x**6 - 5*x - 2 = 0. Только стандартная библиотека Python."""
import argparse
import csv
import math
from decimal import Decimal, localcontext
from pathlib import Path

A, B = 1.4, 1.5
Q = 5 / (6 * (5 * A + 2) ** (5 / 6))
M = 6 * A**5 - 5


def f(x):
    return x**6 - 5*x - 2


def df(x):
    return 6*x**5 - 5


def phi(x):
    return (5*x + 2)**(1/6)


def solve(eps, method, max_iter=1000):
    """Возвращает (корень, историю (k, x, невязка, оценка ошибки))."""
    if not math.isfinite(eps) or eps < 1e-14:
        raise ValueError('Точность должна быть конечным числом не меньше 1e-14.')
    if method not in ('iteration', 'newton'):
        raise ValueError('Неизвестный метод.')
    x = B  
    history = [(0, x, abs(f(x)), abs(f(x))/M)]
    for k in range(1, max_iter + 1):
        next_x = phi(x) if method == 'iteration' else x - f(x)/df(x)
        error = Q/(1-Q)*abs(next_x-x) if method == 'iteration' else abs(f(next_x))/M
        history.append((k, next_x, abs(f(next_x)), error))
        if error <= eps:
            return next_x, history
        x = next_x
    raise RuntimeError('Превышено максимальное число итераций.')


def reference_root():
    """Независимый эталон: бисекция с десятичной арифметикой."""
    with localcontext() as ctx:
        ctx.prec = 70
        a, b = Decimal('1.4'), Decimal('1.5')
        for _ in range(220):
            mid = (a+b)/2
            if mid**6 - 5*mid - 2 > 0:
                b = mid
            else:
                a = mid
        return +(a+b)/2


def actual_error(x, reference):
    with localcontext() as ctx:
        ctx.prec = 70
        return float(abs(Decimal.from_float(x)-reference))


def svg_plot(path, series, xlabel, ylabel, title, log=False):
    """Сохраняет автономный SVG-график без сторонних библиотек."""
    from html import escape
    colors = ['#2563eb', '#dc2626', '#059669']
    def transform(y):
        return math.log10(max(y, 1e-18)) if log else y
    points = [(x, transform(y)) for _, data in series for x, y in data]
    xmin, xmax = min(x for x,y in points), max(x for x,y in points)
    ymin, ymax = min(y for x,y in points), max(y for x,y in points)
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += 1
    def screen(x, y):
        return 85 + 650*(x-xmin)/(xmax-xmin), 420 - 330*(transform(y)-ymin)/(ymax-ymin)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="530" viewBox="0 0 820 530">', '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" font-size="13">', f'<text x="85" y="30" font-size="18">{escape(title)}</text>']
    for i in range(6):
        x = xmin+(xmax-xmin)*i/5
        y = ymin+(ymax-ymin)*i/5
        sx, sy = 85+650*i/5, 420-330*i/5
        parts += [f'<path d="M {sx} 90 V 420 M 85 {sy} H 735" stroke="#e5e7eb"/>', f'<text x="{sx}" y="445" text-anchor="middle">{x:.3g}</text>', f'<text x="75" y="{sy+4}" text-anchor="end">{("10^"+format(y,".2f")) if log else format(y,".3g")}</text>']
    if not log and ymin <= 0 <= ymax:
        _, sy = screen(xmin, 0)
        parts.append(f'<path d="M 85 {sy} H 735" stroke="#555"/>')
    for i, (name, data) in enumerate(series):
        coordinates = ' '.join(f'{sx:.2f},{sy:.2f}' for sx,sy in (screen(x,y) for x,y in data))
        parts += [f'<polyline points="{coordinates}" fill="none" stroke="{colors[i]}" stroke-width="2"/>', f'<text x="{85+i*240}" y="65" fill="{colors[i]}">{escape(name)}</text>']
    parts += [f'<text x="400" y="480" text-anchor="middle">{escape(xlabel)}</text>', f'<text transform="translate(20 270) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>', '</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps', type=float, help='Точность, например 1e-8')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    try:
        eps = args.eps if args.eps is not None else float(input('Введите точность eps (например 1e-8): '))
        results = [(name, *solve(eps, method)) for name, method in [('Простая итерация', 'iteration'), ('Ньютон', 'newton')]]
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    reference = reference_root()
    print(f'Интервал: [{A}, {B}], x0 = {B}; q = {Q:.6f}')
    print(f'Эталонный корень: {reference:.25f}')
    curves = []
    for name, root, history in results:
        print(f'\n{name}: x = {root:.15f}, итераций: {len(history)-1}')
        print(' k       x_k              |f(x_k)|      оценка ошибки   фактическая ошибка')
        rows = [(k,x,res,bound,actual_error(x,reference)) for k,x,res,bound in history]
        for k,x,res,bound,error in rows:
            print(f'{k:2d}  {x:.15f}  {res:.3e}     {bound:.3e}       {error:.3e}')
        filename = 'iteration.csv' if name == 'Простая итерация' else 'newton.csv'
        with (args.output/filename).open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['k', 'x', 'residual', 'error_bound', 'actual_error'])
            writer.writerows(rows)
        curves.append((name, [(k,error) for k,x,res,bound,error in rows]))
    svg_plot(args.output/'errors.svg', curves, 'Количество итераций k', 'Абсолютная погрешность', 'Зависимость погрешности от числа итераций', log=True)
    svg_plot(args.output/'function.svg', [('f(x) = x⁶ − 5x − 2', [(i/500, f(i/500)) for i in range(1001)])], 'x', 'f(x)', 'Графическое определение начального приближения')
    print(f'\nТаблицы и графики сохранены в {args.output.resolve()}')


if __name__ == '__main__':
    main()
