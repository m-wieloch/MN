import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle

from typing import Tuple


# zad1
def polly_A(x: np.ndarray) -> np.ndarray:
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(x, np.ndarray):
        return None

    elif isinstance(x, np.ndarray):
        return P.polyfromroots(x)

    else:
        return None


def roots_20(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(a, np.ndarray):
        return None

    elif isinstance(a, np.ndarray):
        for i in range(0, 20):
            b = a + ((10**(-10)) * np.random.random_sample())
            c = P.polyroots(b)
            result = (b, c)
            return result

    else:
        return None


# zad 2
def frob_a(wsp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(wsp, np.ndarray):
        return None

    elif isinstance(wsp, np.ndarray):

        m = np.zeros((len(wsp), len(wsp)))
        counter = 0

        for i in range(len(wsp)):
            for j in range(len(wsp)):
                if i == len(wsp) - 1:
                    m[i][j] = -wsp[counter]
                    counter = counter + 1

                if i == j and j < len(wsp) - 1:
                    m[i][j + 1] = 1

        result = (m, np.linalg.eigvals(m), scipy.linalg.schur(m), P.polyroots(wsp))

        return result

    else:
        return None
