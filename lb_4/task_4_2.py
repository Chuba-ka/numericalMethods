"""Задание 4.2, вариант 24: краевая задача."""
from math import atan, pi
from methods import Solution, grid, solve_ivp, thomas


def acceleration(x: float, y: float, z: float) -> float:
    return 2*y / (x*x+1)


def exact(x: float) -> float:
    return x*x + x + 1 + (x*x+1)*atan(x)


def shooting(h: float, tolerance: float = 1e-12,
             max_iterations: int = 20) -> Solution:
    """Подбор y(0)=eta методом секущих; y'(0)=2 фиксировано."""
    target = 3 + pi/2

    def shot(eta):
        result = solve_ivp(acceleration, 0, 1, eta, 2, h)
        return result, result.y[-1] - target

    eta0, eta1 = 0.0, 2.0
    _, r0 = shot(eta0)
    result, r1 = shot(eta1)
    for _ in range(max_iterations):
        if abs(r1) <= tolerance:
            return result
        if abs(r1-r0) < 1e-15:
            raise ValueError('Метод секущих: совпали невязки.')
        eta2 = eta1 - r1*(eta1-eta0)/(r1-r0)
        eta0, r0, eta1 = eta1, r1, eta2
        result, r1 = shot(eta1)
    raise ValueError('Метод стрельбы не сошёлся.')


def finite_difference(h: float) -> Solution:
    x = grid(0, 1, h)
    n = len(x)
    lower, diagonal, upper, rhs = ([0.0]*n for _ in range(4))
    # Фиктивный узел: (y1-y_{-1})/(2h)=2,
    # y1-2y0+y_{-1}=2h²y0 => (1+h²)y0-y1=-2h.
    # Это граничное условие второго порядка с учётом ОДУ.
    diagonal[0], upper[0], rhs[0] = 1+h*h, -1, -2*h
    for i in range(1, n-1):
        lower[i], upper[i] = 1, 1
        diagonal[i] = -2 - 2*h*h/(1+x[i]**2)
    diagonal[-1], rhs[-1] = 1, 3+pi/2
    return Solution(x, thomas(lower, diagonal, upper, rhs))


if __name__ == '__main__':
    from main import main
    main(default_task='4.2')
