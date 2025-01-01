"""Пример вычисления линейной зависимости."""

import numpy as np

# Пример данных
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

# Вычисление линейной зависимости
slope, intercept = np.polyfit(x, y, 1)
print("Угловой коэффициент (slope) :", slope)
print("Смещение (intercept) :", intercept)
