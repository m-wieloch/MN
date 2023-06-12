##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple


def chebyshev_nodes(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(n, int):
        return None

    x = np.array([])

    for k in range(0, n + 1):
        x = np.append(x, (np.cos((k*np.pi)/n)))

    return x


def bar_czeb_weights(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(n, int):
        return None

    delta = 0
    w = np.array([])

    for j in range(0, n + 1):
        if j == 0 or j == n:
            delta = 0.5

        if n > j > 0:
            delta = 1

        temp = np.power(-1, j) * delta
        w = np.append(w, temp)

    return w


def barycentric_inte(xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray) or not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray):
        return None

    if np.shape(xi) != np.shape(yi) or np.shape(xi) != np.shape(wi):
        return None

    result = []

    for i in np.nditer(x):
        if i in xi:
            # omijanie dzielenia przez 0
            result.append(yi[np.where(xi == i)[0][0]])
        else:
            laplace = wi / (i - xi)
            result.append(yi @ laplace / sum(laplace))

    return np.array(result)


def L_inf(xr: Union[int, float, List, np.ndarray], x: Union[int, float, List, np.ndarray]) -> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(xr, (int, float, List, np.ndarray)) or not isinstance(x, (int, float, List, np.ndarray)):
        return np.NaN

    if np.shape(xr) != np.shape(x):
        return np.NaN 

    if isinstance(xr, (int, float)) and isinstance(x, (int, float)):
        result = abs(xr - x)
        return result

    if isinstance(xr, (List, np.ndarray)) and isinstance(x, (List, np.ndarray)):

        my_abs = []

        for i in range(len(xr)):
            my_abs.append(abs(xr[i]-x[i]))

        result = max(my_abs)
        return result


# Funkcje z zadania 2

# Funkcja ciągła nieróżniczkowalna
f_1 = lambda x: np.sign(x) * x + np.power(x, 2)

# Funkcja różniczkowalna jednokrotnie
f_2 = lambda x: np.sign(x) * np.power(x, 2)

# Funkcja różniczkowalna trzykrotnie
f_3 = lambda x: np.power(abs(np.sin(5*x)), 3)

# Trzy funkcje analityczne
f_4a1 = lambda x: 1 / (1 + (1 * np.power(x, 2)))

f_4a25 = lambda x:  1 / (1 + (25 * np.power(x, 2)))

f_4a100 = lambda x:  1 / (1 + (100 * np.power(x, 2)))

# Funkcja nieciągła
f_5 = lambda x:  np.sign(x)
