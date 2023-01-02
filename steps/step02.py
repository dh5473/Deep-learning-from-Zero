import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data  # 데이터 꺼내기
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, in_data):
        raise NotImplementedError()  # 이 메서드는 상속해서 구현해야함을 알림


class Square(Function):
    def forward(self, x):
        return x ** 2


x = Variable(np.array(10))
# f = Function()
f = Square()
y = f(x)
print(type(y))  # <class '__main__.Variable'>
print(y.data)  # 100