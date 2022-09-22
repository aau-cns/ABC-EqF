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

from symmetry.symmetry import *


class TestSymmetry(unittest.TestCase):
    """Test the symmetry group"""

    # Test iterations
    iterations = 100

    # Number of elements in list B
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

    def assertListOfSymGroupEqual(self, L1: list, L2: list):
        assert (len(L1) == len(L2))
        for i in range(len(L1)):
            self.assertMatricesEqual(L1[i].as_matrix(), L2[i].as_matrix())

    def assertStateEqual(self, S1: State, S2: State):
        self.assertMatricesEqual(S1.R.as_matrix(), S2.R.as_matrix())
        self.assertMatricesEqual(S1.b, S2.b)
        self.assertListOfSymGroupEqual(S1.S, S2.S)

    def assertSymGroupEqual(self, X1: G, X2: G):
        self.assertMatricesEqual(X1.A.as_matrix(), X2.A.as_matrix())
        self.assertMatricesEqual(X1.a, X2.a)
        self.assertListOfSymGroupEqual(X1.B, X2.B)

    def testG(self):
        self.assertRaises(TypeError, G, np.eye(3))
        self.assertRaises(TypeError, G, SO3.identity(), [1, 2, 3])
        self.assertRaises(TypeError, G, SO3.identity(), np.zeros((3, 1)))
        self.assertRaises(TypeError, G, SO3.identity(), np.zeros((3, 3)), [1, 2, 3])

    def test_associative(self):
        for t in range(self.iterations):
            X1 = G.random(self.n)
            X2 = G.random(self.n)
            X3 = G.random(self.n)
            Z1 = (X1 * X2) * X3
            Z2 = X1 * (X2 * X3)
            self.assertSymGroupEqual(Z1, Z2)

    def test_inverse_identity(self):
        for t in range(self.iterations):
            X = G.random(self.n)
            XInv = X.inv()
            I = G.identity(self.n)
            I1 = X * XInv
            I2 = XInv * X
            self.assertSymGroupEqual(I, I1)
            self.assertSymGroupEqual(I, I2)
            self.assertSymGroupEqual(I1, I2)
            X1 = X * I
            X2 = I * X
            self.assertSymGroupEqual(X, X1)
            self.assertSymGroupEqual(X, X2)
            self.assertSymGroupEqual(X1, X2)
            XInv1 = XInv * I
            XInv2 = I * XInv
            self.assertSymGroupEqual(XInv, XInv1)
            self.assertSymGroupEqual(XInv, XInv2)
            self.assertSymGroupEqual(XInv1, XInv2)

    def test_velocity_action(self):
        for t in range(self.iterations):
            X1 = G.random(self.n)
            X2 = G.random(self.n)
            u = Input.random()
            u0 = velocityAction(G.identity(self.n), u)
            self.assertMatricesEqual(u0.w, u.w)
            u1 = velocityAction(X2, velocityAction(X1, u))
            u2 = velocityAction(X1 * X2, u)
            self.assertMatricesEqual(u1.w, u2.w)

    def test_state_action(self):
        for t in range(self.iterations):
            X1 = G.random(self.n)
            X2 = G.random(self.n)
            xi = State.random(self.n)
            xi0 = stateAction(G.identity(self.n), xi)
            self.assertStateEqual(xi, xi0)
            xi1 = stateAction(X2, stateAction(X1, xi))
            xi2 = stateAction(X1 * X2, xi)
            self.assertStateEqual(xi1, xi1)

if __name__ == "__main__":

    test = TestSymmetry()

    print("*******************************")
    print("Running symmetry test!")
    print("*******************************")
    print("*******************************")
    print("Construction test...")
    print("*******************************")
    test.testG()
    print("*******************************")
    print("Associativity test...")
    print("*******************************")
    test.test_associative()
    print("*******************************")
    print("Inverse test...")
    print("*******************************")
    test.test_inverse_identity()
    print("*******************************")
    print("Velocity action test ...")
    print("*******************************")
    test.test_velocity_action()
    print("*******************************")
    print("State action test ...")
    print("*******************************")
    test.test_state_action()
    print("*******************************")
    print("All tests passed")
    print("*******************************")
