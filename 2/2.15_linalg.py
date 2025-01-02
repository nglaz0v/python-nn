"""Пример вычисления обратной матрицы."""

import numpy as np

# Создание матрицы
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Вычисление обратной матрицы
inverse_matrix = np.linalg.inv(A)
print(inverse_matrix)
