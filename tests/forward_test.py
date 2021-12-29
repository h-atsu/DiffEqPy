from DiffEqPy.utils import numerical_diff
from DiffEqPy import Variable
import DiffEqPy
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.))
        y = x**2
        expected = 4.0
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = x**2
        y.backward()
        expected = np.array(6.)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.random(1))
        y = x**2
        y.backward()
        num_grad = numerical_diff(lambda x: x**2, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
