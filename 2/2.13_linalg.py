"""Пример умножения матрицы признаков на вектор весов."""

import numpy as np

#· Пример данных
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

w = np.array([0.5, 0.25, 0.75])

# Умножение матрицы признаков на вектор весов
result = np.dot(X, w)
print(result)