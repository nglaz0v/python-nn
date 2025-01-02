"""
Пример ReLU.
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)

# Пример входных данных
x = np.array([-1, 2, 3, -4, 0, 5])

# Применение ReLU к входным данным
result = relu(x)
print("Результат после применения ReLU:", result)
