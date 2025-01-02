"""
Пример плотности вероятности непрерывных величин.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Генерируем выборку из нормального распределения
data = np.random.normal(loc=0, scale=1, size=1000)

# Строим гистограмму выборки
plt.hist(data, bins=30, density=True, alpha=0.5, color='b')

# Вычисляем среднее и стандартное отклонение выборки
mean = np.mean(data)
std_dev = np.std(data)

# Вычисляем плотность вероятности нормального распределения для данной выборки
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
р = norm.pdf(x, mean, std_dev)

# Строим график плотности вероятности нормального распределения
plt.plot(x, р, 'k', linewidth=2)
title = "Fit results: mean %.2f, std dev = %.2f" % (mean, std_dev)
plt.title(title)
plt.show()
