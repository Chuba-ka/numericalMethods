"""Задание 4.1, вариант 24: задача Коши."""
from math import e, exp
from methods import solve_ivp


def acceleration(x: float, y: float, z: float) -> float:
    return (y - (x+1)*z) / x**2


def exact(x: float) -> float:
    return x + 1 + x*exp(1/x)


def solve(h: float, method: str = 'rk4'):
    return solve_ivp(acceleration, 1, 2, 2+e, 1, h, method)


if __name__ == '__main__':
    from main import main
    main(default_task='4.1')
