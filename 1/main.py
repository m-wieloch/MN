import math
import numpy as np


def cylinder_area(r: float, h: float):
    """Obliczenie pola powierzchni walca.
    Szczegółowy opis w zadaniu 1.

    Parameters:
    r (float): promień podstawy walca
    h (float): wysokosć walca

    Returns:
    float: pole powierzchni walca
    """
    if r <= 0 or h <= 0:
        return math.nan
    else:
        pp: float = 2 * math.pi * r * r
        pb: float = 2 * math.pi * r * h
        return pp + pb


def fib(n: int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if n <= 0:
        return None
    if not isinstance(n, int):
        return None
    if n == 1:
        return 1
    else:
        i, first, second = 0, 0, 1
        fibonacci = []
        while i < n:
            first, second = second, first + second
            fibonacci.append(first)
            i = i + 1
        return [fibonacci]


def matrix_calculations(a: float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    M = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    if a == 0:
        Minv = math.nan
    else:
        Minv = np.linalg.inv(M)
    Mt = np.transpose(M)
    Mdet = np.linalg.det(M)
    result = (Minv, Mt, Mdet)
    return result


def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if m < 0 or n < 0:
        return None
    if not isinstance(m, int):
        return None
    if not isinstance(n, int):
        return None

    M = np.zeros((m, n))
    i, j = 0, 0

    while i < m:
        j = 0
        while j < n:
            if j > i:
                M[i][j] = j
            elif i > j:
                M[i][j] = i
            elif j == i:
                M[i][j] = j
            j = j + 1
        i = i + 1
    return M

