"""
Пример уравнения метода наименьших квадратов.
"""

import numpy as np

# Исходные данные
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
у = np.dot(X, np.array([1, 2])) + 3

# Применение МНК для решения задачи линейной регрессии
coefficients = np.linalg.lstsq(X, у, rcond=None)[0]
print("Коэффициенты регрессии:", coefficients)
