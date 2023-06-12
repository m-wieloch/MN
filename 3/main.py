import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.

    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona

    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(v, (int, float, List, np.ndarray)) and isinstance(v_aprox, (int, float, List, np.ndarray)):

        # v = np.ndarray

        if isinstance(v, np.ndarray) and isinstance(v_aprox, np.ndarray):
            if all((i == j) or (i == 1) or (j == 1) for i, j in zip(v.shape[::-1], v_aprox.shape[::-1])):
                result = abs(v - v_aprox)
                return result

            else:
                return np.NaN

        elif isinstance(v, np.ndarray) and isinstance(v_aprox, List) and v.shape[0] == len(v_aprox):
            result = np.zeros(len(v_aprox))

            for i in range(len(v_aprox)):
                result[i] = abs(v[i] - v_aprox[i])

            return result

        elif isinstance(v, np.ndarray) and isinstance(v_aprox, List) and v.shape[1] == len(v_aprox):
            result = np.zeros(len(v_aprox))

            for i in range(len(v_aprox)):
                result[i] = abs(v_aprox[i] - v[0][1])

        elif isinstance(v, np.ndarray) and isinstance(v_aprox, List) and v.shape[0] != len(v_aprox) and v.shape[1] != len(v_aprox):
            return np.NaN

        elif isinstance(v, np.ndarray) and isinstance(v_aprox, (int, float)) or isinstance(v_aprox, np.ndarray) and isinstance(v, (int, float)):
            result = abs(v - v_aprox)
            return result

        # v = int, float

        elif isinstance(v, (int, float)) and isinstance(v_aprox, (int, float)):
            result = abs(v - v_aprox)
            return result

        elif isinstance(v, (int, float)) and isinstance(v_aprox, list):
            result = np.zeros(len(v_aprox))

            for i in range(len(v_aprox)):
                result[i] = abs(v - v_aprox[i])

            return result

        elif isinstance(v_aprox, (int, float)) and isinstance(v, list):
            result = np.zeros(len(v))

            for i in range(len(v)):
                result[i] = abs(v_aprox - v[i])

            return result

        # v = List

        elif isinstance(v, List) and isinstance(v_aprox, List) and len(v) == len(v_aprox) and not isinstance(v[0], List):
            result = np.zeros(len(v))

            for i in range(len(v)):
                result[i] = abs(v[i] - v_aprox[i])

            return result

        elif isinstance(v, List) and isinstance(v_aprox, List) and len(v) == len(v_aprox) and isinstance(v[0], List):
            result = np.zeros((len(v), len(v[0])))

            for i in range(len(v)):
                for j in range(len(v[0])):
                    result[i][j] = abs((v[i])[j] - (v_aprox[i])[j])

            return result

        elif isinstance(v, List) and isinstance(v_aprox, List) and len(v) != len(v_aprox):
            return np.NaN

        elif isinstance(v, List) and isinstance(v_aprox, np.ndarray) and len(v) == v_aprox.shape[0]:
            result = np.zeros(len(v))

            for i in range(len(v)):
                result[i] = abs(v[i] - v_aprox[i])

            return result

        elif isinstance(v, List) and isinstance(v_aprox, np.ndarray) and len(v) != v_aprox.shape[0]:
            return np.NaN

    else:
        return np.NaN


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    abs_error = absolut_error(v, v_aprox)

    if abs_error is np.NaN:
        return np.NaN

    elif isinstance(v, np.ndarray):
        result = abs_error / v
        return result

    elif isinstance(v, (int, float)) and v == 0:
        return np.NaN

    elif isinstance(v, np.ndarray) and not v.any():
        return np.NaN

    elif isinstance(abs_error, np.ndarray) and isinstance(v, List):
        result = np.zeros(len(v))

        for i in range(len(v)):
            if v[i] == 0:
                return np.NaN
            result[i] = abs_error[i] / v[i]

        return result

    else:
        result = abs_error / v
        return result


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) and isinstance(c, (int, float)):
        b = np.power(2, n)

        p1 = b - b + c
        p2 = b + c - b

        result = abs(p1 - p2)
        return result

    else:
        return np.NaN


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(x, (int, float)) and isinstance(n, int) and n > 0:
        result = 0

        for i in range(n):
            result += (1 / (scipy.math.factorial(i))) * (np.power(x, i))

        return result

    else:
        return np.NaN


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) and isinstance(x, (int, float)):
        if x == 0 or k == 0:
            return np.cos(0)

        if k == 1:
            return np.cos(x)

        if k > 1:
            m = k - 1
            return 2 * np.cos(x) * coskx1(m, x) - coskx1(m-1, x)

        if k < 0:
            return np.NaN

    else:
        return np.NaN


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k, int) and isinstance(x, (int, float)):
        if x == 0 or k == 0:
            result = (np.cos(0), np.sin(0))
            return result

        if k == 1:
            result = (np.cos(x), np.sin(x))
            return result

        if k > 1:
            m = k - 1

            rec = coskx2(m, x)
            res_cos = np.cos(x) * rec[0] - np.sin(x) * rec[1]
            res_sin = np.sin(x) * rec[0] + np.cos(x) * rec[1]

            result = (res_cos, res_sin)
            return result

        if k < 0:
            return np.NaN

    else:
        return np.NaN


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n, int) and n >= 1:

        result = 0

        for i in range(1, n + 1):
            result += (1 / (np.power(i, 2)))

        result = np.sqrt(6*result)

        return result

    else:
        return np.NaN
