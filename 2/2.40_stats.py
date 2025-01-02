"""
Пример взаимодействия дискретной и непрерывной случайных величин.
"""

import numpy as np

# Дискретная случайная величина
X_discrete = np.array([1, 2, 3, 4, 5])  # значения
P_discrete = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # вероятности

# Непрерывная случайная величина
X_continuous = np.linspace(0, 10, 1000)  # значения
f_continuous = np.exp(-0.5* (X_continuous - 5)**2) / np.sqrt(2 * np.pi)  # плотность

# Вычисление математических ожиданий
E_discrete = np.sum(X_discrete * P_discrete)
E_continuous = np.trapz(X_continuous * f_continuous, X_continuous)

print("Математическое ожидание дискретной случайной величины:", E_discrete)
print("Математическое ожидание непрерывной случайной величины:", E_continuous)
