"""
Пример функции Leaky ReLU.
"""

import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha*x)

# Пример входных данных
x = np.array([-1, 2, 3, -4, 0, 5])

# Применение Leaky ReLU к входным данным
result = leaky_relu(x)
print("Результат после применения Leaky ReLU:", result)
