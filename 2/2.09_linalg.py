"""Пример обратной матрицы."""

import numpy as np

A = np.array([[1, 2], [3, 4]])  # исходная матрица

# Обратная матрица
A_inv = np.linalg.inv(A)

print(A_inv)
