from typing import ForwardRef, List
import numpy as np
import weakref
from memory_profiler import profile
import contextlib

import DiffEqPy


class Config:
    # only have class attribute
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supporsed".format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    # 関数だがあたかもインスタンス変数として振る舞う
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return DiffEqPy.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return DiffEqPy.functions.transpose(self, axes)

    @property
    def T(self):
        return DiffEqPy.functions.transpose(self)

    def sum(self, axis=None, keepdims=None):
        return DiffEqPy.functions.sum(self, axis, keepdims)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        # ' '*9は 'variable('の分空文字挿入する役割
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        # 自分の親関数を取得
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            # 親関数の入力に対する勾配を計算
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            # 逆伝播計算する時にグラフを残すか消すか
            with using_config('enable_backprop', create_graph):
                # backward計算するときはVariableの演算が呼ばれる
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
                # f.inputに勾配を流しきったのでf.outputの勾配を保持する必要がない
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        # xsとしてndarrayを作る
        xs = [x.data for x in inputs]
        # 基底Fuctionにover rideされたforwardメソッドが呼び出される, forwardはnumpy計算
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # code for backpropagation
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            # 複数値の返り値それぞれにcreatorをself記憶させる
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # self.outputs = [output for output in outputs] memory efficiency
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Parameter(Variable):
    pass


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = DiffEqPy.functions.sum_to(gx0, self.x0_shape)
            gx1 = DiffEqPy.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    return Add()(x0, as_array(x1))


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    return Sub()(x0, as_array(x1))


def rsub(x0, x1):
    return Sub()(as_array(x1), x0)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1,   gy * x0


def mul(x0, x1):
    return Mul()(x0, as_array(x1))


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1


def div(x0, x1):
    return Div()(x0, as_array(x1))


def rdiv(x0, x1):
    return Div()(as_array(x1), x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x**(c-1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
