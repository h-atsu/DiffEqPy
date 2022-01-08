from DiffEqPy import Variable
from DiffEqPy.functions import *
from DiffEqPy.utils import *
from DiffEqPy.layers import *
from DiffEqPy.models import MLP
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)
x = np.random.rand(100,1)
#y = 5 + 2*x + np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)/10


model = MLP((10,10,1))


t = np.linspace(0,1).reshape(-1,1)


y_pred = model(t)
plt.plot(np.linspace(0,1), y_pred.data, c='r')
plt.scatter(x,y)
#plt.ylim(y.min()-0.1,y.max()+0.1)
#plt.xlim(x.min()-0.1,x.max()+0.1)


lr = 0.1
loss_his = []
for i in range(20000):
    y_pred = model(x)
    loss = mean_squared_error(y_pred, y)
    loss_his.append(loss.data)
    model.cleargrads()    
    loss.backward()
    for p in model.params():
        p.data -= lr * p.grad.data


plt.plot(loss_his)


plot_dot_graph(loss)


y_pred = model(t)


plt.plot(np.linspace(0,1), y_pred.data, c='r')
plt.scatter(x,y)
#plt.ylim(y.min()-0.1,y.max()+0.1)
#plt.xlim(x.min()-0.1,x.max()+0.1)
