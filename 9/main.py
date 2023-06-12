import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle
from inspect import isfunction

from typing import Union, List, Tuple


def fun(x):
    return np.exp(-2 * x) + x ** 2 - 1


def dfun(x):
    return -2 * np.exp(-2 * x) + 2 * x


def ddfun(x):
    return 4 * np.exp(-2 * x) + 2


def bisection(a: Union[int, float], b: Union[int, float], f: typing.Callable[[float], float], epsilon: float,
              iteration: int) -> Tuple[float, int]:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and isfunction(f) and
            isinstance(epsilon, float) and isinstance(iteration, int)):
        return None

    elif f(a) * f(b) >= 0:
        return None

    elif a < b:
        if f(a) < 0 and f(b) > 0 or f(a) > 0 and f(b) < 0:
            actual_iter = 0

            while actual_iter <= iteration:
                m = (a + b) / 2

                if abs(f(m)) < epsilon:
                    result = (m, actual_iter)
                    break

                elif f(a) * f(m) > 0:
                    a = m

                else:
                    b = m

                actual_iter = actual_iter + 1

            if actual_iter == iteration:
                result = (m, actual_iter)

            return result
    else:
        return None


def secant(a: Union[int, float], b: Union[int, float], f: typing.Callable[[float], float], epsilon: float,
           iteration: int) -> Tuple[float, int]:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and isfunction(f) and
            isinstance(epsilon, float) and isinstance(iteration, int)):
        return None

    elif f(a) * f(b) >= 0:
        return None

    elif a < b:
        actual_iter = 0

        while actual_iter <= iteration:
            y = a - f(a) * ((b - a) / (f(b) - f(a)))

            if f(y) - epsilon <= 0 and f(y) + epsilon >= 0:
                result = (y, actual_iter)
                return result

            elif f(a) * f(y) < 0:
                b = y

            elif f(b) * f(y) < 0:
                a = y

            if actual_iter == iteration:
                result = (y, actual_iter)
                return result

            actual_iter = actual_iter + 1
    else:
        return None


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float],
           ddf: typing.Callable[[float], float], a: Union[int, float], b: Union[int, float], epsilon: float,
           iteration: int) -> Tuple[float, int]:
    """ Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry:
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and isfunction(f) and isfunction(df)
            and isfunction(ddf) and isinstance(epsilon, float) and isinstance(iteration, int)):
        return None

    elif f(a) * f(b) >= 0:
        return None

    elif a < b:
        actual_iter = 1 # 1 ponieważ: Spodziewany wynik: (0.9165625831056982, 4), aktualny (0.9165625831056982, 3)
        rb = b
        while actual_iter < iteration:
            y = b - f(b) / df(b)

            if f(y) - epsilon <= 0 and f(y) + epsilon >= 0:
                if rb >= y >= a:
                    result = (y, actual_iter)
                    return result
                else:
                    return None
            b = y

            if actual_iter == iteration:
                result = (y, actual_iter)
                return result

            actual_iter = actual_iter + 1

    else:
        return None
