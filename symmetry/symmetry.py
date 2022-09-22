#!/usr/bin/env python3

"""Definition of the Symmetry group"""

__author__ = "Alessandro Fornasier"
__copyright__ = "Copyright (C) 2022 Alessandro Fornasier"
__credits__ = ["Alessandro Fornasier"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Alessandro Fornasier"
__email__ = "alessandro.fornasier@ieee.org"
__status__ = "Academic research"

#
# External libraries:
# - pylie : https://github.com/pvangoor/pylie
#
import pdb

from system.system import Direction, State, Input
from utils.utils import *
from pylie import SO3
from typing import List

"""Coordinate representation
----------
Possible values are:
"EXPONENTIAL":    for exponential coordinates
"NORMAL":         for normal coordinates
----------
"""
coordinate = "EXPONENTIAL"

class G:
    """Symmetry group (SO(3) |x so(3)) x SO(3) x ... x SO(3)
    ----------
    Each element of the B list is associated with a calibration states in State's S list where the association is done
    via corresponding index. In general B[i] is the SO(3) element of the symmetry group that correspond to the
    state's calibration state S[i]. For example, let's assume we want to use three known direction a, b, and c, where
    only the sensor that measure b is uncalibrated (we'd like to estimate the calibration states). Therefore,
    the system's d list is defined as d = [b, a, c], and the state's S list is defined as S = [Sb]. The symmetry group
    B list should be defined as B = [Bb] where Ba is the SO(3) element of the symmetry group that is related to Sb
    ----------
    """

    A: SO3
    a: np.ndarray
    B: List[SO3]

    def __init__(self, A: SO3 = SO3.identity(), a: np.ndarray = np.zeros((3, 3)), B: List[SO3] = None):
        """Initialize the symmetry group G

        :param A: SO3 element
        :param a: np.ndarray with shape (3, 3) corresponding to a skew symmetric matrix
        :param B: list of SO3 elements
        """

        if not isinstance(A, SO3):
            raise TypeError("A has to be of type SO3")
        self.A = A
        if not (isinstance(a, np.ndarray) and a.shape == (3, 3)):
            raise TypeError("a has to be a numpy array with shape (3, 3)")
        self.a = a
        if B is None:
            self.B = []
        else:
            for b in B:
                if not isinstance(b, SO3):
                    raise TypeError("Elements of B have to be of type SO3")
            self.B = B

    def __mul__(self, other: 'G') -> 'G':
        """Define the group operation

        :param other: G
        :return: A element of the group G given by the "multiplication" of self and other
        """

        assert (isinstance(other, G))
        assert (len(self.B) == len(other.B))
        return G(self.A * other.A,
                 self.a + SO3.wedge(self.A.as_matrix() @ SO3.vee(other.a)),
                 [self.B[i]*other.B[i] for i in range(len(self.B))])

    @staticmethod
    def identity(n : int):
        """Return the identity of the symmetry group with n elements of SO3 related to sensor calibration states

        :param n: number of elements in list B associated with calibration states
        :return: The identity of the group G
        """

        return G(SO3.identity(), np.zeros((3, 3)), [SO3.identity() for _ in range(n)])

    @staticmethod
    def random(n: int):
        """Return a random element of the symmetry group with n elements of SO3 related to sensor calibration states

        :param n: number of elements in list B associated with calibration states
        :return: A random element of the group G
        """

        return G(SO3.exp(np.random.randn(3, 1)),
                 SO3.wedge(np.random.randn(3, 1)),
                 [SO3.exp(np.random.randn(3, 1)) for _ in range(n)])

    def inv(self) -> 'G':
        """Return the inverse element of the symmetry group

        :return: A element of the group G given by the inverse of self
        """

        return G(self.A.inv(), -SO3.wedge(self.A.inv().as_matrix() @ SO3.vee(self.a)), [B.inv() for B in self.B])

    def exp(x: np.ndarray) -> 'G':
        """Return a group element X given by X = exp(x) where x is a numpy array

        :param x: A numpy array
        :return: A element of the group G given by the exponential of x
        """

        if not (isinstance(x, np.ndarray) and x.shape[0] >= 6 and x.shape[1] == 1):
            raise ValueError("Wrong shape, a numpy array with shape (3n, 1) has to be provided")
        if (x.size % 3) != 0:
            raise ValueError("Wrong size, a numpy array with size multiple of 3 has to be provided")

        n = int((x.size - 6) / 3)
        A = SO3.exp(x[0:3, :])
        a = SO3.wedge(SO3LeftJacobian(x[0:3, :]) @ x[3:6, :])
        B = [SO3.exp(x[(6 + 3 * i):(9 + 3 * i)]) for i in range(n)]

        return G(A, a, B)

    def log(X: 'G') -> np.ndarray:
        """Return a numpy array x given by x = log(X) where X is a group element

        :param X: A element of the group G
        :return: A numpy array given by the logarithm of X
        """

        if not isinstance(X, G):
            raise TypeError

        n = len(X.B)
        x = np.zeros(((6 + 3 * n), 1))
        x[0:3, :] = SO3.log(X.A)
        x[3:6, :] = np.linalg.inv(SO3LeftJacobian(x[0:3, :])) @ SO3.vee(X.a)
        x[6:, :] = np.asarray([SO3.log(B) for B in X.B]).reshape((3 * n), 1)

        return x


def SO3LeftJacobian(arr: np.ndarray) -> np.ndarray:
    """Return the SO(3) Left Jacobian

    :param arr: A numpy array with shape (3, 1)
    :return: The left Jacobian of SO(3)
    """

    if not (isinstance(arr, np.ndarray) and arr.shape == (3, 1)):
        raise ValueError("A numpy array with shape (3, 1) has to be provided")

    angle = np.linalg.norm(arr)

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(angle, 0.):
        return np.eye(3) + 0.5 * SO3.wedge(arr)

    axis = arr / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return (s / angle) * np.eye(3) + \
           (1 - (s / angle)) * np.outer(axis, axis) + \
           ((1 - c) / angle) * SO3.wedge(axis)


def stateAction(X: G, xi: State) -> State:
    """Action of the symmetry group on the state space, return phi(X, xi) (Equation 4)

    :param X: A element of the group G
    :param xi: A element of the State
    :return: A new element of the state given by the action of phi of G in the State space
    """

    if len(xi.S) != len(X.B):
        raise ValueError("the number of calibration states and B elements of the symmetry group has to match")

    return State(xi.R * X.A,
                 X.A.inv().as_matrix() @ (xi.b - SO3.vee(X.a)),
                 [SO3.from_matrix(X.A.inv().as_matrix() @ xi.S[i].as_matrix() @ X.B[i].as_matrix()) for i in range(len(X.B))])


def velocityAction(X: G, u: Input) -> Input:
    """Action of the symmetry group on the input space, return psi(X, u) (Equation 5)

    :param X: A element of the group G
    :param u: A element of the Input
    :return: A new element of the Input given by the action of psi of G in the Input space
    """

    return Input(X.A.inv().as_matrix() @ (u.w - SO3.vee(X.a)), u.Sigma)


def outputAction(X: G, y: Direction, idx: int = -1) -> np.ndarray:
    """Action of the symmetry group on the output space, return rho(X, y) (Equation 6)

    :param X: A element of the group G
    :param y: A direction measurement
    :param idx: indicate the index of the B element in the list, -1 in case no B element exist
    :return: A numpy array given by the action of rho of G in the Output space
    """

    if idx == -1:
        return X.A.inv().as_matrix() @ y.d
    else:
        return X.B[idx].inv().as_matrix() @ y.d


def local_coords(e: State) -> np.ndarray:
    """Local coordinates assuming __xi_0 = identity (Equation 9)

    :param e: A element of the State representing the equivariant error
    :return: Local coordinates assuming __xi_0 = identity
    """

    if coordinate == "EXPONENTIAL":
        tmp = [SO3.log(S) for S in e.S]
        eps = np.vstack((SO3.log(e.R), e.b, np.asarray(tmp).reshape((3 * len(tmp)), 1)))
    elif coordinate == "NORMAL":
        raise ValueError("Normal coordinate representation is not implemented yet")
        # X = G(e.R, -SO3.wedge(e.R @ e.b), e.S)
        # eps = G.log(X)
    else:
        raise ValueError("Invalid coordinate representation")

    return eps


def local_coords_inv(eps: np.ndarray) -> "State":
    """Local coordinates inverse assuming __xi_0 = identity

    :param eps: A numpy array representing the equivariant error in local coordinates
    :return: Local coordinates inverse assuming __xi_0 = identity
    """

    X = G.exp(eps)
    if coordinate == "EXPONENTIAL":
        e = State(X.A, eps[3:6, :], X.B)
    elif coordinate == "NORMAL":
        raise ValueError("Normal coordinate representation is not implemented yet")
        # stateAction(X, State(SO3.identity(), np.zeros((3, 1)), [SO3.identity() for _ in range(len(X.B))]))
    else:
        raise ValueError("Invalid coordinate representation")

    return e


def stateActionDiff(xi: State) -> np.ndarray:
    """Differential of the phi action phi(xi, E) at E = Id in local coordinates (can be found within equation 23)

    :param xi: A element of the State
    :return: (Dtheta) * (Dphi(xi, E) at E = Id)
    """
    coordsAction = lambda U: local_coords(stateAction(G.exp(U), xi))
    differential = numericalDifferential(coordsAction, np.zeros(((6 + 3 * len(xi.S)), 1)))
    return differential
