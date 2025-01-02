"""
Пример гиперболического тангенса.
"""

import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

# Генерация данных для построения графика
x_values = np.linspace(-10, 10, 100)
y_values = tanh(x_values)

# Визуализация гиперболического тангенса
plt.plot(x_values, y_values, 'r-', label='Гиперболический тангенс')
plt.title('Гиперболический тангенс')
plt.xlabel('х')
plt.ylabel('tanh(x) ')
plt.grid(True)
plt.legend()
plt.show()
