from DiffEqPy import Variable, Model
from DiffEqPy.models import MLP
import numpy as np
import math
from DiffEqPy.utils import plot_dot_graph
import DiffEqPy.functions as F
from DiffEqPy.layers import *
import matplotlib.pyplot as plt


def sphere(x, y):
    return x**2 + y**2


def matyas(x, y):
    return (1 + (x+y+1)**2*(19-14*x+3*x**2 - 14*y + 6*x*y + 3*y*y)) * (30+(2*x-3*y)**2*(18-32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))


def my_sin(x, threshold=1e-150):
    y = 0
    for i in range(1000):
        c = (-1) ** i / math.factorial(2*i+1)  # coefficient of ith poly
        t = c*x ** (2*i + 1)
        y = y+t
        if abs(t.data) < threshold:
            break
    return y


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


x = Variable(np.random.randn(5, 10), name='x')
mynet = MLP((10, 1))
mynet.plot(x)
print('h')
