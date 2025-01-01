"""
Пример диапазона значений непрерывных величин.
"""

import numpy as np

# Создаем диапазон значений от О до 10 с шагом 0.1
start = 0
end = 10
step = 0.1

# Генерируем диапазон значений
continuous_values = [value for value in np.arange(start, end, step)]

# Выводим полученный диапазон значений
print(continuous_values)
