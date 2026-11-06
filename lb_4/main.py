"""Запуск лабораторной работы №4, вариант 24."""
import argparse
import csv
from pathlib import Path
from methods import error_rows
import task_4_1
import task_4_2

HEADERS = ('x', 'y_h', 'y_exact', 'error_h', 'y_h2', 'error_h2',
           'runge_romberg_h2', 'y_richardson')


def main(default_task: str = 'all') -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['all', '4.1', '4.2'], default=default_task)
    parser.add_argument('--h', type=float, default=0.1, help='шаг сетки (по умолчанию 0.1)')
    parser.add_argument('--output', type=Path, help='каталог для таблиц CSV')
    args = parser.parse_args()
    jobs = []
    if args.task in ('all', '4.1'):
        for method, name, p in [('euler', 'Эйлер', 1), ('rk4', 'Рунге–Кутта', 4),
                                ('adams', 'Адамс', 4)]:
            jobs.append(('4_1_'+method, name, p, task_4_1.exact,
                         lambda h, m=method: task_4_1.solve(h, m)))
    if args.task in ('all', '4.2'):
        jobs.extend([('4_2_shooting', 'Стрельба', 4, task_4_2.exact, task_4_2.shooting),
                     ('4_2_fdm', 'Конечные разности', 2, task_4_2.exact,
                      task_4_2.finite_difference)])
    try:
        results = [(key, name, p, error_rows(solver(args.h), solver(args.h/2), exact, p))
                   for key, name, p, exact, solver in jobs]
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    for key, name, p, rows in results:
        print(f'\n{key}: {name}, p={p}, h={args.h:g}')
        print('    x         y(h)        y точное     |ошибка h|    |ошибка h/2|   Р–Р (h/2)')
        for x, yh, yt, eh, yf, ef, rr, refined in rows:
            print(f'{x:6.3f}  {yh:12.8f}  {yt:12.8f}  {eh:12.4e}  {ef:12.4e}  {rr:12.4e}')
        print(f'max |ошибка h| = {max(r[3] for r in rows):.6e}; '
              f'max Р–Р (h/2) = {max(r[6] for r in rows):.6e}')
        if args.output:
            with (args.output / (key+'.csv')).open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(HEADERS)
                writer.writerows(rows)


if __name__ == '__main__':
    main()
