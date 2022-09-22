#!/usr/bin/env python3

"""Utilities"""

__author__ = "Alessandro Fornasier"
__copyright__ = "Copyright (C) 2022 Alessandro Fornasier"
__credits__ = ["Alessandro Fornasier"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Alessandro Fornasier"
__email__ = "alessandro.fornasier@ieee.org"
__status__ = "Academic research"


import numpy as np


def numericalDifferential(f, x) -> np.ndarray:
    """Compute the numerical derivative via central difference"""

    if isinstance(x, float):
        x = np.reshape([x], (1, 1))
    h = 1e-6
    fx = f(x)
    n = fx.shape[0]
    m = x.shape[0]
    Df = np.zeros((n, m))
    for j in range(m):
        ej = np.zeros((m, 1))
        ej[j, 0] = 1.0
        Df[:, j:j+1] = (f(x + h * ej) - f(x - h * ej)) / (2*h)
    return Df


def blockDiag(A : np.ndarray, B : np.ndarray) -> np.ndarray:
    """Create a lock diagonal matrix from blocks A and B

    :param A: numpy array
    :param B: numpy array
    :return: numpy array representing a block diagonal matrix composed of blocks A and B
    """

    if A is None:
        return B
    elif B is None:
        return A
    else:
        return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],[np.zeros((B.shape[0], A.shape[1])), B]])


def repBlock(A : np.ndarray, n: int) -> np.ndarray:
    """Create a block diagonal matrix repeating the A block n times

    :param A: numpy array representing the block A
    :param n: number of times to repeat A
    :return: numpy array representing a block diagonal matrix composed of n-times the blocks A
    """

    res = None
    for _ in range(n):
        res = blockDiag(res, A)
    return res


def checkNorm(x: np.ndarray, tol: float = 1e-3):
    """Check norm of a vector being 1 or nan

    :param x: A numpy array
    :param tol: tollerance, default 1e-3
    :return: Boolean true if norm is 1 or nan
    """
    return abs(np.linalg.norm(x) - 1) < tol or np.isnan(np.linalg.norm(x))
