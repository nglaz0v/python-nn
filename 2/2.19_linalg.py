"""Пример уравнения диагональной матрицы и единичного вектора."""

import numpy as np

# Создаём диагональную матрицу
diagonal_matrix = np.diag([3, 5, 7])

# Создаём единичный вектор
unit_vector = np.array([1, 1, 1])

# Умножаем диагональную матрицу на единичный вектор
result = np.dot(diagonal_matrix, unit_vector)
print("Результат умножения:", result)
