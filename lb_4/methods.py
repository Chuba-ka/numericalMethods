"""Численные методы для ОДУ второго порядка, без сторонних библиотек."""
from dataclasses import dataclass
import math
from typing import Callable

Acceleration = Callable[[float, float, float], float]


@dataclass
class Solution:
    x: list[float]
    y: list[float]
    z: list[float] | None = None


def grid(a: float, b: float, h: float) -> list[float]:
    if not math.isfinite(h) or h <= 0:
        raise ValueError('Шаг должен быть положительным конечным числом.')
    ratio = (b - a) / h
    n = round(ratio)
    if n < 1 or not math.isclose(ratio, n, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Шаг должен делить длину отрезка на целое число частей.')
    return [a + (b - a) * i / n for i in range(n + 1)]


def rk4_step(f: Acceleration, x: float, y: float, z: float,
             h: float) -> tuple[float, float]:
    k1, l1 = h * z, h * f(x, y, z)
    k2 = h * (z + l1 / 2)
    l2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)
    k3 = h * (z + l2 / 2)
    l3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)
    k4 = h * (z + l3)
    l4 = h * f(x + h, y + k3, z + l3)
    return (y + (k1 + 2*k2 + 2*k3 + k4) / 6,
            z + (l1 + 2*l2 + 2*l3 + l4) / 6)


def solve_ivp(f: Acceleration, a: float, b: float, y0: float,
              z0: float, h: float, method: str = 'rk4') -> Solution:
    """Система y'=z, z'=f(x,y,z). Адамс — явный AB4 со стартом RK4."""
    if method not in ('euler', 'rk4', 'adams'):
        raise ValueError(f'Неизвестный метод: {method}')
    x = grid(a, b, h)
    if method == 'adams' and len(x) < 5:
        raise ValueError('Для Адамса нужно не менее четырёх шагов.')
    y, z = [y0], [z0]
    for i in range(len(x) - 1):
        if method == 'euler':
            yn, zn = y[i] + h*z[i], z[i] + h*f(x[i], y[i], z[i])
        elif method == 'rk4' or i < 3:
            yn, zn = rk4_step(f, x[i], y[i], z[i], h)
        else:
            weights = (55, -59, 37, -9)
            yn = y[i] + h/24 * sum(w*z[i-j] for j, w in enumerate(weights))
            zn = z[i] + h/24 * sum(
                w*f(x[i-j], y[i-j], z[i-j]) for j, w in enumerate(weights))
        y.append(yn)
        z.append(zn)
    return Solution(x, y, z)


def thomas(lower: list[float], diagonal: list[float], upper: list[float],
           rhs: list[float]) -> list[float]:
    """Метод прогонки; исходные массивы не изменяются."""
    d, r = diagonal.copy(), rhs.copy()
    for i in range(len(d)):
        if abs(d[i]) < 1e-14:
            raise ValueError('Нулевой ведущий элемент в методе прогонки.')
        if i + 1 < len(d):
            factor = lower[i+1] / d[i]
            d[i+1] -= factor * upper[i]
            r[i+1] -= factor * r[i]
    y = [0.0] * len(d)
    y[-1] = r[-1] / d[-1]
    for i in range(len(d)-2, -1, -1):
        y[i] = (r[i] - upper[i]*y[i+1]) / d[i]
    return y


def error_rows(coarse: Solution, fine: Solution, exact: Callable[[float], float],
               order: int) -> list[tuple[float, ...]]:
    """Рунге–Ромберг: оценка ошибки решения h/2 в общих узлах."""
    if len(fine.x) != 2*len(coarse.x)-1:
        raise ValueError('Для оценки нужны сетки с шагами h и h/2.')
    rows = []
    for i, x in enumerate(coarse.x):
        yh, yf, yt = coarse.y[i], fine.y[2*i], exact(x)
        correction = (yf-yh) / (2**order-1)
        rows.append((x, yh, yt, abs(yh-yt), yf, abs(yf-yt),
                     abs(correction), yf+correction))
    return rows
