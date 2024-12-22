#!/usr/bin/env python3

"""Implementation of Attitude-Bias-Calibration EqF form:
"Overcoming Bias: Equivariant Filter Design for Biased Attitude Estimation with Online Calibration"
https://ieeexplore.ieee.org/document/9905914
"""

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

from symmetry.symmetry import *
from system.system import Direction, Measurement
from scipy.linalg import expm

def lift(xi: State, u: Input) -> np.ndarray:
    """The Lift of the system (Theorem 3.8, Equation 7)

    :param xi: A element of the State
    :param u: A element of the Input space
    :return: A numpy array representing the Lift
    """

    n = len(xi.S)
    L = np.zeros(6 + 3 * n)
    L[0:3] = (u.w - xi.b)
    L[3:6] = -u.W() @ xi.b
    for i in range(n):
        L[(6 + 3 * i):(9 + 3 * i)] = xi.S[i].inv().as_matrix() @ L[0:3]

    return L


class EqF:

    def __init__(self, Sigma: np.ndarray, n: int, m: int):
        """Initialize EqF

        :param Sigma: Initial covariance
        :param n: Number of calibration states
        :param m: Total number of available sensor
        """

        self.__dof = 6 + 3 * n
        self.__n_cal = n
        self.__n_sensor = m

        if not (isinstance(Sigma, np.ndarray) and (Sigma.shape[0] == Sigma.shape[1] == self.__dof)):
            raise TypeError(f"Initial covariance has to be provided as a numpy array with shape ({self.__dof}, {self.__dof})")
        if not np.all(np.linalg.eigvals(Sigma) >= 0):
            raise TypeError("Covariance matrix has to be semi-positive definite")
        if not (isinstance(n, int) and n >= 0):
            raise TypeError("Number of calibration state has to be unsigned")
        if not (isinstance(m, int) and m > 1):
            raise TypeError("Number of direction sensor has to be grater-equal than 2")

        self.__X_hat = G.identity(n)
        self.__Sigma = Sigma
        self.__xi_0 = State.identity(n)
        self.__Dphi0 = stateActionDiff(self.__xi_0)             # Within equation 23
        self.__InnovationLift = np.linalg.pinv(self.__Dphi0)    # Within equation 23

    def stateEstimate(self) -> State:
        """Return estimated state

        :return: Estimated state
        """
        return stateAction(self.__X_hat, self.__xi_0)

    def propagation(self, u: Input, dt: float):
        """Propagate the filter state

        :param u: Angular velocity measurement from IMU
        :param dt: delta time between timestamp of last propagation/update and timestamp of angular velocity measurement
        """

        if not isinstance(u, Input):
            raise TypeError("angular velocity measurement has to be provided as a Input element")

        L = lift(self.stateEstimate(), u)                                                           # Equation 7

        Phi_DT = self.__stateTransitionMatrix(u, dt)                                                # Equation 17
        # Equivalent
        # A0t = self.__stateMatrixA(u)                                                              # Equation 14a
        # Phi_DT = expm(A0t * dt)

        Bt = self.__inputMatrixBt()                                                                 # Equation 27
        M_DT = (Bt @ blockDiag(u.Sigma, repBlock(1e-9 * np.eye(3), self.__n_cal)) @ Bt.T) * dt

        self.__X_hat = self.__X_hat * G.exp(L * dt)                                                 # Equation 18
        self.__Sigma = Phi_DT @ self.__Sigma @ Phi_DT.T + M_DT                                      # Equation 19

    def update(self, y: Measurement):
        """Update the filter state

        :param y: A Direction measurement
        """

        # Cross-check calibration
        assert (y.cal_idx <= self.__n_cal)

        Ct = self.__measurementMatrixC(y.d, y.cal_idx)                              # Equation 14b
        delta = SO3.wedge(y.d.d) @ outputAction(self.__X_hat.inv(), y.y, y.cal_idx)
        Dt = self.__outputMatrixDt(y.cal_idx)
        S = Ct @ self.__Sigma @ Ct.T + Dt @ y.Sigma @ Dt.T                          # Equation 21
        K = self.__Sigma @ Ct.T @ np.linalg.inv(S)                                  # Equation 22
        Delta = self.__InnovationLift @ K @ delta                                   # Equation 23
        self.__X_hat = G.exp(Delta) * self.__X_hat                                  # Equation 24
        self.__Sigma = (np.eye(self.__dof) - K @ Ct) @ self.__Sigma                 # Equation 25

    def __stateMatrixA(self, u: Input) -> np.ndarray:
        """Return the state matrix A0t (Equation 14a)

        :param u: Input
        :return: numpy array representing the state matrix A0t
        """

        W0 = velocityAction(self.__X_hat.inv(), u).W()
        A1 = np.zeros((6, 6))

        if coordinate == "EXPONENTIAL":
            A1[0:3, 3:6] = -np.eye(3)
            A1[3:6, 3:6] = W0
            A2 = repBlock(W0, self.__n_cal)
        elif coordinate == "NORMAL":
            raise ValueError("Normal coordinate representation is not implemented yet")
        else:
            raise ValueError("Invalid coordinate representation")

        return blockDiag(A1, A2)

    def __stateTransitionMatrix(self, u: Input, dt: float) -> np.ndarray:
        """Return the state transition matrix Phi (Equation 17)

        :param u: Input
        :param dt: Delta time
        :return: numpy array representing the state transition matrix Phi
        """

        W0 = velocityAction(self.__X_hat.inv(), u).W()
        Phi1 = np.zeros((6, 6))
        Phi12 = -dt * (np.eye(3) + (dt / 2) * W0 + ((dt**2) / 6) * W0 * W0)
        Phi22 = np.eye(3) + dt * W0 + ((dt**2) / 2) * W0 * W0

        if coordinate == "EXPONENTIAL":
            Phi1[0:3, 0:3] = np.eye(3)
            Phi1[0:3, 3:6] = Phi12
            Phi1[3:6, 3:6] = Phi22
            Phi2 = repBlock(Phi22, self.__n_cal)
        elif coordinate == "NORMAL":
            raise ValueError("Normal coordinate representation is not implemented yet")
        else:
            raise ValueError("Invalid coordinate representation")

        return blockDiag(Phi1, Phi2)

    def __inputMatrixBt(self) -> np.ndarray:
        """Return the Input matrix Bt

        :return: numpy array representing the state matrix Bt
        """

        if coordinate == "EXPONENTIAL":
            B1 = blockDiag(self.__X_hat.A.as_matrix(), self.__X_hat.A.as_matrix())
            B2 = None
            for B in self.__X_hat.B:
                B2 = blockDiag(B2, B.as_matrix())
        elif coordinate == "NORMAL":
            raise ValueError("Normal coordinate representation is not implemented yet")
        else:
            raise ValueError("Invalid coordinate representation")

        return blockDiag(B1, B2)

    def __measurementMatrixC(self, d: Direction, idx: int) -> np.ndarray:
        """Return the measurement matrix C0 (Equation 14b)

        :param d: Known direction
        :param idx: index of the related calibration state
        :return: numpy array representing the measurement matrix C0
        """

        Cc = np.zeros((3, 3 * self.__n_cal))

        # If the measurement is related to a sensor that has a calibration state
        if idx >= 0:
            Cc[(3 * idx):(3 + 3 * idx), :] = SO3.wedge(d.d)

        return SO3.wedge(d.d) @ np.hstack((SO3.wedge(d.d), np.zeros((3, 3)), Cc))

    def __outputMatrixDt(self, idx: int) -> np.ndarray:
        """Return the measurement output matrix Dt

        :param idx: index of the related calibration state
        :return: numpy array representing the output matrix Dt
        """

        # If the measurement is related to a sensor that has a calibration state
        if idx >= 0:
            return self.__X_hat.B[idx].as_matrix()
        else:
            return self.__X_hat.A.as_matrix()
