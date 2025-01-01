"""
Пример биноминального распределения.
"""

import numpy as np
import matplotlib.pyplot as plt

# Параметры биноминального распределения
n = 10  # количество испытаний
р = 0.5  # вероятность успеха

# Генерация данных с биноминальным распределением
binomial_data = np.random.binomial(n, р, 1000)

# Визуализация данных
plt.hist(binomial_data, bins=11, density=True, alpha=0.6, color='b')
plt.title('Биноминальное распределение')
plt.xlabel('Количество успехов в серии испытаний')
plt.ylabel('Частота')
plt.show()
