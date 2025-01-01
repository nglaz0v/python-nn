"""Пример уравнения спектрального (сингулярного) разложения матрицы."""

import numpy as np

# Создание произвольной матрицы
A = np.array([[1, 2],
              [3, 4]])

# Сингулярное разложение
U, Sigma, Vt = np.linalg.svd(A)

# Проверка результатов
reconstructed_A = np.dot(U, np.dot(np.diag(Sigma), Vt))
print("Исходная матрица A:\n", A)
print("Восстановленная матрица A:\n", reconstructed_A)
