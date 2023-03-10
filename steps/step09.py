import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)  # Create on a single line


def exp(x):
    return Exp()(x)


x = Variable(np.array(0.5))
# a = square(x)
# b = exp(a)
# y = square(b)
y = square(exp(square(x)))  # 연속하여 적용
#  y.grad = np.array(1.0)  # backward 메서드에 추가
y.backward()
print(x.grad)  # 3.297442541400256


x = Variable(np.array(1.0))  # OK
x = Variable(None)  # OK
# x = Variable(1.0)  # NG: 오류 발생


# numpy의 독특한 관례
x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)  # <class 'numpy.ndarray'> 0
print(type(y))  # <class 'numpy.float64'>