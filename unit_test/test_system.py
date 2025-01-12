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

import sys
import os
import unittest

# Update path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/unit_test')[0])

from system.system import *


class TestSymmetry(unittest.TestCase):
    """Test the System"""

    # Number of calibration states
    n = 3

    def setUp(self) -> None:
        np.random.seed(0)
        return super().setUp()

    def assertMatricesEqual(self, M1: np.ndarray, M2: np.ndarray):
        assert (M1.shape == M2.shape)
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                self.assertAlmostEqual(M1[i, j], M2[i, j])

    def assertListOfMatricesEqual(self, L1: list, L2: list):
        assert (len(L1) == len(L2))
        for i in range(len(L1)):
            self.assertMatricesEqual(L1[i], L2[i])

    def assertStateEqual(self, S1: State, S2: State):
        self.assertMatricesEqual(S1.R.as_matrix(), S2.R.as_matrix())
        self.assertMatricesEqual(S1.b, S2.b)
        self.assertListOfMatricesEqual(S1.S, S2.S)

    def testDirection(self):
        self.assertRaises(TypeError, Direction, 5)
        self.assertRaises(TypeError, Direction, np.array([1, 1, 0]).reshape(3, 1))
        Direction(np.array([1, 0, 0]))

    def testInput(self):
        self.assertRaises(TypeError, Input, 5, np.eye(3))
        self.assertRaises(TypeError, Input, np.array([1, 0, 0]), np.eye(3))
        self.assertRaises(TypeError, Input, np.array([1, 0, 0]), np.zeros((3, 2)))
        Input(np.array([1, 1, 1]), np.eye(6))

    def testState(self):
        self.assertRaises(TypeError, State, np.eye(3), np.zeros(3), [1, 2, 3])
        self.assertRaises(TypeError, State, SO3.identity(), [1, 0, 0], [1, 2, 3])
        self.assertRaises(TypeError, State, SO3.identity(), np.zeros((1, 3)), [1, 2, 3])
        self.assertRaises(TypeError, State, SO3.identity(), np.zeros(3), [1, 2, 3])
        self.assertRaises(TypeError, State, SO3.identity(), np.zeros(3), np.eye(3))
        State(SO3.identity(), np.zeros(3), [SO3.identity()])

    def testMeasurement(self):
        self.assertRaises(TypeError, Measurement, [1, 0, 0], [1, 0, 0], np.eye(3))
        self.assertRaises(TypeError, Measurement, np.zeros((1, 3)), np.zeros(3), np.eye(3))
        self.assertRaises(TypeError, Measurement, np.zeros(3), np.zeros((1, 3)), np.eye(3))
        self.assertRaises(TypeError, Measurement, np.zeros(3), np.zeros(3), np.eye(3), -3)
        self.assertRaises(TypeError, Measurement, np.zeros(3), np.zeros(3), np.eye(3), 3.2)
        Measurement(np.array([1, 0, 0]), np.array([1, 0, 0]), np.eye(3), 1)


if __name__ == "__main__":

    test = TestSymmetry()

    print("*******************************")
    print("Running system test!")
    print("*******************************")
    print("*******************************")
    print("Test Direction Construction...")
    print("*******************************")
    test.testDirection()
    print("*******************************")
    print("Test Input Construction...")
    print("*******************************")
    test.testInput()
    print("*******************************")
    print("Test State Construction...")
    print("*******************************")
    test.testState()
    print("*******************************")
    print("Test Measurement Construction...")
    print("*******************************")
    test.testMeasurement()
    print("*******************************")
    print("All tests passed")
    print("*******************************")