from DiffEqPy import Variable, Model
from DiffEqPy.models import MLP
import numpy as np
import math
from DiffEqPy.utils import plot_dot_graph
import DiffEqPy.functions as F
from DiffEqPy.layers import *
import matplotlib.pyplot as plt
from DiffEqPy.optimizers import SGD
from DiffEqPy.datasets import *


train_set = Spiral(train=True)
print(train_set[0])
print(len(train_set))
