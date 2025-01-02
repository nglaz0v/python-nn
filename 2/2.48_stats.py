"""
Пример сигмоидной функции.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Генерация данных для построения графика
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

# Визуализация сигмоидной функции
plt.plot(x_values, y_values, 'b-', label='Сигмоидная функция')
plt.title('Сигмоидная функция')
plt.xlabel('х')
plt.ylabel('sigmoid (х)')
plt.grid(True)
plt.legend()
plt.show()
