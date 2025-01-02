"""
Пример функции распределения.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Определяем параметры нормального распределения
mu = 0
sigma = 1

# Создаём массив значений х
x = np.linspace(-5, 5, 1000)

# Вычисляем функцию нормального распределения
cdf = norm.cdf(x, mu, sigma)

# Строим график функции распределения
plt.plot(x, cdf, label='CDF')
plt.title('Cumulative Distribution Function (CDF) of Normal Distribution')
plt.xlabel('х')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()
