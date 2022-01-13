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


def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.
    return x


batch_size = 100
max_epoch = 10
hidden_size = 1000

train_set = MNIST(train = True, transform=f)
test_set = MNIST(train = False, transform=f)
train_loader = DataLoader(train_set , batch_size)
test_loader = DataLoader(test_set , batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size,10))
#model = MLP((hidden_size, hidden_size,10), activation=F.relu)
optimizer = SGD().setup(model)


x = np.array([val[0] for val in train_set])
t = np.array([val[1] for val in train_set])


ret = {}
ret["train_loss"] = []
ret["test_loss"] = []
ret["train_accuracy"] = []
ret["test_accuracy"] = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0    
    
    for x,t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
        
    print("epoch get_ipython().run_line_magic("d", " \" % (epoch+1))")
    print("train loss: {:.4f}, accuracy: {:.4f}".format(sum_loss / len(train_set), sum_acc / len(train_set)))
    ret["train_loss"].append(sum_loss / len(train_set))
    ret["train_accuracy"].append(sum_acc / len(train_set))
        
    sum_loss, sum_acc = 0,0
    with DiffEqPy.no_grad():
        for x,t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y,t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print("test loss: {:.4f}, accuracy: {:.4f}".format(sum_loss / len(test_set), sum_acc / len(test_set)))
    ret["test_loss"].append(sum_loss / len(test_set))
    ret["test_accuracy"].append(sum_acc / len(test_set))


plt.plot(ret["train_loss"])
plt.plot(ret["test_loss"])


plt.plot(ret["train_accuracy"])
plt.plot(ret["test_accuracy"])


model.plot(x)
