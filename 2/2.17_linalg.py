"""Пример вычисления линейной оболочки."""

import numpy as np

# Пример данных
points = np.array([[1, 1],
                   [2, 3],
                   [4, 5]])

# Вычисление линейной оболочки
hull = np.linalg.qr(points)[0]
print(hull)
