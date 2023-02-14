if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)  # variable([11 12 13])

y.backward()
print(x0.grad)  # variable([1 1 1])
print(x1.grad)  # variable([3])

y = x0 * x1
print(y)  # variable([10 20 30])

x0.cleargrad()
x1.cleargrad()
y.backward()
print(x0.grad)  # variable([10 10 10])
print(x1.grad)  # variable([6])