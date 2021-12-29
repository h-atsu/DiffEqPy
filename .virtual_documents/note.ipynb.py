from DiffEqPy import Variable
from DiffEqPy.functions import *
from DiffEqPy.utils import *
from DiffEqPy.layers import *
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)
x = np.random.rand(100,1)
#y = 5 + 2*x + np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)


l1 = Linear(10)
l2 = Linear(1)


def predict(x):
    y = l1(x)
    y = sigmoid(y)
    y = l2(y)
    return y


t = np.linspace(0,1).reshape(-1,1)


y_pred = predict(t)
plt.plot(np.linspace(0,1), y_pred.data, c='r')
plt.scatter(x,y)
#plt.ylim(y.min()-0.1,y.max()+0.1)
#plt.xlim(x.min()-0.1,x.max()+0.1)


lr = 0.2
loss_his = []
for i in range(10000):
    y_pred = predict(x)
    loss = mean_squared_error(y_pred, y)    
    loss_his.append(loss.data)
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data


plt.plot(loss_his)


plot_dot_graph(loss)


y_pred = predict(t)


plt.plot(np.linspace(0,1), y_pred.data, c='r')
plt.scatter(x,y)
#plt.ylim(y.min()-0.1,y.max()+0.1)
#plt.xlim(x.min()-0.1,x.max()+0.1)
