import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)  # 1.0

x.data = np.array(2.0)
print(x.data)  # 2.0

# 차원 확인하기
x1 = np.array(1)
x2 = np.array([1, 2, 3])
x3 = np.array([[1, 2, 3],
               [4, 5, 6]])
print(x1.ndim)  # 0
print(x2.ndim)  # 1
print(x3.ndim)  # 2
