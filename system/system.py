#!/usr/bin/env python3

"""Definition of the Biased Attitude System"""

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

import numpy as np
from dataclasses import dataclass
from pylie import SO3
from typing import List
from utils.utils import checkNorm

class Direction:
    """Define a direction as a S2 element"""

    # Direction
    d: np.ndarray
    
    def __init__(self, d: np.ndarray):
        """Initialize direction

        :param d: A numpy array with shape (3, 1) and norm 1 representing the direction
        """

        if not (isinstance(d, np.ndarray) and d.shape == (3, 1) and checkNorm(d)):
            raise TypeError("Direction has to be provided as a (3, 1) vector")
        self.d = d


class State:
    """Define the state of the Biased Attitude System
    ----------
    R is a rotation matrix representing the attitude of the body
    b is a 3-vector representing the gyroscope bias
    S is a list of rotation matrix, each representing the calibration of the corresponding direction sensor
    ----------
    Let's assume we want to use three known direction a, b, and c, where only the sensor that measure b is
    uncalibrated (we'd like to estimate the calibration states). Therefore, the system's d list looks like
    d = [b, a, c], and the S list should look like S = [Sb]. The association between d and S is done via indeces.
    In general S[i] correspond to the calibration state of the sensor that measure the direcion d[i]
    ----------
    """

    # Attitude rotation matrix R
    R: SO3

    # Gyroscope bias b
    b: np.ndarray

    # Sensor calibrations S
    S:  List[SO3]

    def __init__(self, R : SO3 = SO3.identity(), b : np.ndarray = np.zeros((3, 1)), S: List[SO3] = None):
        """Initialize State

        :param R: A SO3 element representing the attitude of the system as a rotation matrix
        :param b: A numpy array with shape (3, 1) representing the gyroscope bias
        :param S: A list of SO3 elements representing the calibration states for "uncalibrated" sensors,
        if no sensor require a calibration state, than S will be initialized as an empty list
        """

        if not isinstance(R, SO3):
            raise TypeError("the attitude rotation matrix R has to be of type SO3")
        self.R = R

        if not (isinstance(b, np.ndarray) and b.shape == (3, 1)):
            raise TypeError("The gyroscope bias has to be probvided as numpy array with shape (3, 1)")
        self.b = b

        if S is None:
            self.S = []
        else:
            if not isinstance(S, list):
                raise TypeError("Calibration states has to be provided as a list")
            for calibration in S:
                if not isinstance(calibration, SO3):
                    raise TypeError("Elements of the list of calibration states have to be of type SO3")
            self.S = S

    @staticmethod
    def identity(n: int):
        """Return a identity state  with n calibration states

        :param n: number of elements in list B associated with calibration states
        :return: The identity element of the State
        """

        return State(SO3.identity(),
                     np.zeros((3, 1)),
                     [SO3.identity() for _ in range(n)])

    @staticmethod
    def random(n: int) -> 'State':
        """Return a random state with n calibration states

        :param n: number of elements in list S associated with calibration states
        :return: A random element of the State
        """

        return State(SO3.exp(np.random.randn(3, 1)),
                     np.random.randn(3, 1),
                     [SO3.exp(np.random.randn(3, 1)) for _ in range(n)])

    def as_dict(self) -> dict:
        """Return the State as dictionary

        :return: self as dictionary
        """

        return {"R": self.R, "b": self.b, "S": self.S}


# class System:
#     """Define the iased Attitude System
#     ----------
#     d is a list of directions known to the system
#     state is the state of the system
#     ----------
#     The list d is ordered with directions where the corresponding sensor is uncalibrated first and calibrated after.
#     For example, let's assume we want to use three known direction a, b, and c, where only the sensor that measure b is
#     uncalibrated. Therefore, the d list looks like d = [b, a, c],
#     ----------
#     """
#
#     # Known directions
#     d: List[Direction]
#
#     # State
#     state: State
#
#     def __init__(self, d: List[Direction], state: State = State()):
#         """
#         Initialize State
#
#         :param d: A list of Direction
#         :param state: A State element
#         """
#
#         if not isinstance(d, list):
#             raise TypeError("Directions has to be provided as a list")
#         if len(d) < 2:
#             raise ValueError("At least two known directions have to be provided")
#         for direction in d:
#             if not isinstance(direction, Direction):
#                 raise TypeError("Elements of the list of directions have to be of type Direction")
#         self.d = d
#
#         if not isinstance(state, State):
#             raise TypeError("the state of the system has to be of type State")
#         self.state = state
#
#     def as_dict(self) -> dict:
#         """Return the System as dictionary
#
#         :return: self as a dictionary
#         """
#
#         return {**self.state.as_dict(), "d": [dirs.d for dirs in self.d]}


class Input:
    """Define the input space of the Biased Attitude System
    ----------
    w is a 3-vector representing the angular velocity measured by a gyroscope
    ----------
    """

    # Angular velocity
    w: np.ndarray

    # Noise covariance of angular velocity
    Sigma: np.ndarray

    def __init__(self, w: np.ndarray, Sigma: np.ndarray):
        """Initialize Input

        :param w: A numpy array with shape (3, 1) representing the angular velocity measurement from a gyroscope
        :param Sigma: A numpy array with shape (6, 6) representing the noise covariance of the
        angular velocity measurement and gyro bias random walk
        """

        if not (isinstance(w, np.ndarray) and w.shape == (3, 1)):
            raise TypeError("Angular velocity has to be provided as a numpy array with shape (3, 1)")
        if not (isinstance(Sigma, np.ndarray) and Sigma.shape[0] == Sigma.shape[1] == 6):
            raise TypeError("Input measurement noise covariance has to be provided as a numpy array with shape (6. 6)")
        if not np.all(np.linalg.eigvals(Sigma) >= 0):
            raise TypeError("Covariance matrix has to be semi-positive definite")

        self.w = w
        self.Sigma = Sigma

    @staticmethod
    def random() -> 'Input':
        """Return a random angular velocity

        :return: A random angular velocity as a Input element
        """

        return Input(np.random.randn(3, 1), np.eye(6))

    def W(self) -> np.ndarray:
        """Return the Input as a skew-symmetric matrix

        :return: self.w as a skew-symmetric matrix
        """

        return SO3.wedge(self.w)

class Measurement:
    """Define a measurement
    ----------
    cal_idx is a index corresponding to the cal_idx-th calibration related to the measurement. Let's consider the case
    of 2 uncalibrated sensor with two associated calibration state in State.S = [S0, S1], and a single calibrated sensor.
    cal_idx = 0 indicates a measurement coming from the sensor that has calibration S0, cal_idx = 1 indicates a
    measurement coming from the sensor that has calibration S1. v = -1 indicates that the measurement is coming
    from a calibrated sensor
    ----------
    """

    # measurement
    y: Direction

    # Known direction in global frame
    d: Direction

    # Covariance matrix of the measurement
    Sigma: np.ndarray

    # Calibration index
    cal_idx: int = -1

    def __init__(self, y: np.ndarray, d: np.ndarray, Sigma: np.ndarray, i: int = -1):
        """Initialize measurement

        :param y: A numpy array with shape (3, 1) and norm 1 representing the direction measurement in the sensor frame
        :param d: A numpy array with shape (3, 1) and norm 1 representing the direction in the global frame
        :param Sigma: A numpy array with shape (3, 3) representing the noise covariance of the direction measurement
        :param i: index of the corresponding calibration state
        """

        if not (isinstance(y, np.ndarray) and y.shape == (3, 1) and checkNorm(y)):
            raise TypeError("Measurement has to be provided as a (3, 1) vector")
        if not (isinstance(d, np.ndarray) and d.shape == (3, 1) and checkNorm(d)):
            raise TypeError("Direction has to be provided as a (3, 1) vector")
        if not (isinstance(Sigma, np.ndarray) and Sigma.shape[0] == Sigma.shape[1] == 3):
            raise TypeError("Direction measurement noise covariance has to be provided as a numpy array with shape (3. 3)")
        if not np.all(np.linalg.eigvals(Sigma) >= 0):
            raise TypeError("Covariance matrix has to be semi-positive definite")
        if not (isinstance(i, int) or i == -1 or i > 0):
            raise TypeError("calibration index is a positive integer or -1")

        self.y = Direction(y)
        self.d = Direction(d)
        self.Sigma = Sigma
        self.cal_idx = i
