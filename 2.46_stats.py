"""
Пример экспоненциального распределения.
"""

import numpy as np
import matplotlib.pyplot as plt

# Параметр экспоненциального распределения (обратное значение среднего)
beta = 0.5

# Генерация данных с экспоненциальным распределением
exponential_data = np.random.exponential(scale=1/beta, size=1000)

# Визуализация данных
plt.hist(exponential_data, bins=30, density=True, alpha=0.6, color='r')
plt.title('Экспоненциальное распределение')
plt.xlabel('Значение')
plt.ylabel('Чacтoтa')
plt.show()
