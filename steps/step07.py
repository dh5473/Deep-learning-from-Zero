import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 함수 가져오기
        if f is not None:
            x = f.input  # 함수 입력 가져오기
            x.grad = f.backward(self.grad)  # 함수의 backward 메서드 호출
            x.backward()  # 하나 앞 변수의 backward 메서드 호출(재귀)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 출력 변수에 창조자 설정
        self.input = input
        self.output = output  # 출력 저장
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        return NotImplementedError()


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


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올라가기
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# 역전파
y.grad = np.array(1.0)

C = y.creator  # 1. 함수 가져오기
b = C.input  # 2. 함수 입력 가져오기
b.grad = C.backward(y.grad)  # 함수의 backward 메서드 호출

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)  # 3.297442541400256


# 새로운 Variable로 역전파 자동 실행
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파
y.grad = np.array(1.0)
y.backward()
print(x.grad)  # 3.297442541400256