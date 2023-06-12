import numpy as np
import pickle

from typing import Union, List, Tuple


def random_matrix_Ab(m: int) -> np.ndarray:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(100, size=(m, m))
        b = np.random.randint(100, size=m)

        return A, b

    else:
        return None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray)  -> (float):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania 
      x: wektor x (m.) zawierający rozwiązania równania 
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów"""
    # None
    # typ
    if not isinstance(A, np.ndarray):
        return None

    if not isinstance(x, np.ndarray):
        return None

    if not isinstance(b, np.ndarray):
        return None

    # wymiar
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        return None

    if len(x.shape) != 1 or x.shape[0] == 0 or x.shape[0] != A.shape[0]:
        return None

    if len(b.shape) != 1 or b.shape[0] == 0 or b.shape[0] != A.shape[0]:
        return None

    r = b - np.matmul(A, x)
    result = np.linalg.norm(r)

    return result


def log_sing_value(n: int, min_order: Union[int, float], max_order: Union[int, float]) -> np.ndarray:
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
         Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
         """
    if not (isinstance(n, int) and isinstance(min_order, (int, float)) and isinstance(max_order, (int, float))):
        return None

    if n <= 0 or min_order > max_order:
        return None

    else:
        result = np.logspace(max_order, min_order, n)
        return result


def order_sing_value(n: int, order: Union[int, float] = 2, site: str = 'gre') -> np.ndarray:
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10. 
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
        """
    if not (isinstance(n, int) and isinstance(order, (int, float)) and isinstance(site, str)):
        return None

    if n <= 0:
        return None

    else:
        random_1 = np.random.rand(1) * 10
        random_2 = np.random.rand(1) * 10

        if random_1 > random_2:
            start = random_1
            stop = random_2

        else:
            start = random_2
            stop = random_1

        vector = np.logspace(start, stop, n)
        vector.shape = (n,)

        if order > 0:
            if site == 'low':
                vector[-1] = vector[-1] / 10**order

            elif site == 'gre':
                vector[0] = vector[0] * 10**order

            else:
                return None

        elif order < 0:
            if site == 'low':
                vector[-1] = vector[-1] * 10**order

            elif site == 'gre':
                vector[0] = vector[0] / 10**order

            else:
                return None

        elif order == 0:
            return None

        return vector


def create_matrix_from_A(A: np.ndarray, sing_value: np.ndarray) -> np.ndarray:
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych

            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)


            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """
    if not (isinstance(A, np.ndarray) and isinstance(sing_value, np.ndarray)):
        return None

    if all((m == n) or (n == 1) for m, n in zip(A.shape[::-1], sing_value.shape[::-1])):
        u, s, v = np.linalg.svd(A)
        A = np.dot(u * sing_value, v)
        return A

    else:
        return None
