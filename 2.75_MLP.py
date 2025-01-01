"""
Пример применения глубокого многослойного перцептрона с двумя скрытыми слоями
и сигмоидной функцией.
"""

import numpy as np

# Входные данные
input_data = np.array([0.1, 0.2, 0.3])

# Веса для скрытых слоёв
hidden_weights1 = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]])
hidden_weights2 = np.array([[0.2, 0.3],
                            [0.5, 0.6]])

# Веса для выходного слоя
output_weights = np.array([0.4, 0.7])

# Пересчёт значений скрытых слоёв
hidden_layer1_values = np.dot(input_data, hidden_weights1.T)
hidden_activation1 = 1 / (1 + np.exp(-hidden_layer1_values))
hidden_layer2_values = np.dot(hidden_activation1, hidden_weights2.T)
hidden_activation2 = 1 / (1 + np.exp(-hidden_layer2_values))

# Пересчёт значений выходного слоя
output = np.dot(hidden_activation2, output_weights)

print("Результат:", output)
