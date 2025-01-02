"""
Пример нормального (Гауссовского) распределения.
"""

import numpy as np
import matplotlib.pyplot as plt

# Генерация данных с нормальным распределением
mean = 0  # Среднее значение
std_dev = 1  # Стандартное отклонение
num_samples = 1000

# Генерация данных
normal_data = np.random.normal(mean, std_dev, num_samples)

# Визуализация данных
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='g')
plt.title('Hopмaльнoe (Гауссовское) распределение')
plt.xlabel('Значение')
plt.ylabel('Чacтoтa')
plt.show()
