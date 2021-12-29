import unittest
import numpy as np
from DiffEqPy import Variable
from DiffEqPy.utils import numerical_diff
import DiffEqPy.functions as F


def array_equal(xs, ys):
    return all([x == y for x, y in zip(xs, ys)])


class TestAD(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 * x1
        res = y.data
        expected = np.array([1, 4, 9])
        self.assertTrue(array_equal(res, expected))


class TestDiv(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 / x1
        res = y.data
        expected = np.array([1, 1, 1])
        self.assertTrue(array_equal(res, expected))
