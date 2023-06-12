import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional
from numpy import linalg


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not (isinstance(m, int) and m > 0):
        return None

    else:
        A = np.random.randint(10, size=(m, m))
        b = np.random.randint(10, size=m)

        sum_row = 0
        sum_col = 0

        for i in range(m):
            sum_row = sum_row + np.sum(A[i, :])
            sum_col = sum_col + np.sum(A[:, i])

            sum_row = sum_row - A[i, i]
            sum_col = sum_col - A[i, i]

            if sum_row < sum_col:
                A[i, i] = sum_col + 1
            else:
                A[i, i] = sum_row + 1

        result = (A, b)
        return result


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, np.ndarray):
        return None

    elif not (len(A.shape) == 2 and A.shape[0] == A.shape[1]):
        return None

    else:
        m_diag = np.diag(np.abs(A))
        m_sum = np.sum(np.abs(A), axis=1) - m_diag

        if np.all(m_diag > m_sum):
            return True
        else:
            return False


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not (isinstance(m, int) and m > 0):
        return None

    else:
        A = np.random.randint(10, size=(m, m))
        b = np.random.randint(10, size=m)

        for r in range(m):
            diag = False
            for c in range(m):
                if r == c:
                    diag = True

                if diag:
                    A[c][r] = A[r][c]

        result = (A, b)
        return result


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, np.ndarray):
        return None

    elif not (len(A.shape) == 2 and A.shape[0] == A.shape[1]):
        return None

    else:
        symetric = True

        for r in range(A.shape[0]):
            diag = False
            for c in range(A.shape[0]):
                if r == c:
                    diag = True

                if diag:
                    if A[c][r] != A[r][c]:
                        symetric = False
                        break

            if not symetric:
                break

        return symetric


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    # sprawdzanie typow
    if not (isinstance(A, np.ndarray) and isinstance(b, np.ndarray) and isinstance(x_init, np.ndarray) and
            (isinstance(epsilon, float) or epsilon is None) and (isinstance(maxiter, int) or maxiter is None)):
        return None

    # sprawdzanie wymiarow
    elif not (len(A.shape) == 2 and A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0] and b.shape == x_init.shape):
        return None

    # poprawnosc iteracji i epsilonu
    elif not (maxiter >= 1 and epsilon > 0):
        return None

    # metoda z wykładu
    else:
        D = np.diag(np.diag(A))
        LU = A - D
        x = x_init
        D_inv = np.diag(1 / np.diag(D))
        resid = []

        for i in range(maxiter):
            x_new = np.dot(D_inv, b - np.dot(LU, x))
            r_norm = np.linalg.norm(x_new - x)
            resid.append(r_norm)

            if r_norm < epsilon:
                return x_new, resid
            x = x_new

        return x, resid


# Funkcja random_matrix_Ab z lab6
def random_matrix_Ab(m: int) -> np.ndarray:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        A = np.random.randint(10, size=(m, m))
        b = np.random.randint(10, size=m)

        return A, b

    else:
        return None


# Funkcja sprawdzajaca norme potrzebna do zadan 5-7
def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      x: wektor x (n,) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów
      """
    if not (isinstance(A, np.ndarray) and isinstance(x, np.ndarray) and isinstance(b, np.ndarray)):
        return None
    elif np.size(A, 1) == len(x) and np.size(A, 0) == len(b):
        return linalg.norm(b - A @ x)
    else:
        return None
