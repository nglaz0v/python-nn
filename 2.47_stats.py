"""
Пример геометрического распределения.
"""

import numpy as np
import matplotlib.pyplot as plt

# Вероятность успеха (параметр геометрического распределения)
p = 0.3

# Генерация данных с геометрическим распределением
geometric_data = np.random.geometric(p, size=1000)

# Визуализация данных
plt.hist(geometric_data, bins=20, density=True, alpha=0.6, color='m')
plt.title('Геометрическое распределение')
plt.xlabel('Количество испытаний до первого успеха')
plt.ylabel('Частота')
plt.show()
