from DiffEqPy import Variable, DataLoader
from DiffEqPy.functions import *
from DiffEqPy.utils import *
from DiffEqPy.layers import *
from DiffEqPy.models import MLP
from DiffEqPy.optimizers import SGD
from DiffEqPy.datasets import *
import DiffEqPy
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.
    return x


batch_size = 100
max_epoch = 5
hidden_size = 1000

train_set = MNIST(train=True, transform=f)
test_set = MNIST(train=False, transform=f)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
#model = MLP((hidden_size, hidden_size,10), activation=F.relu)
optimizer = SGD(1e-8).setup(model)

if DiffEqPy.cuda.gpu_enable:
    train_loader.to_gpu()
    test_loader.to_gpu()
    model.to_gpu()

ret = {}
ret["train_loss"] = []
ret["test_loss"] = []
ret["train_accuracy"] = []
ret["test_accuracy"] = []

for epoch in range(max_epoch):
    start = time.time()
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    end = time.time()
    print("epoch: {}, time: {}".format(epoch + 1, end - start))
    print("train loss: {:.4f}, accuracy: {:.4f}".format(
        sum_loss / len(train_set), sum_acc / len(train_set)))
    ret["train_loss"].append(sum_loss / len(train_set))
    ret["train_accuracy"].append(sum_acc / len(train_set))

    sum_loss, sum_acc = 0, 0
    with DiffEqPy.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print("test loss: {:.4f}, accuracy: {:.4f}".format(
        sum_loss / len(test_set), sum_acc / len(test_set)))
    ret["test_loss"].append(sum_loss / len(test_set))
    ret["test_accuracy"].append(sum_acc / len(test_set))
