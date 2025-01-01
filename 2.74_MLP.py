"""
Пример применения многослойного перцептрона с одним скрытым слоем и сигмоидной
функцией.
"""

import numpy as np

# Входные данные
input_data = np.array([0.1, 0.2, 0.3])

# Веса для скрытого слоя
hidden_weights = np.array([[0.4, 0.5, 0.6],
                           [0.7, 0.8, 0.9]])

# Веса для выходного слоя
output_weights = np.array([0.5, 0.6])

# Пересчёт значений скрытого слоя
hidden_layer_values = np.dot(input_data, hidden_weights.T)

# Применение функции активации (например, сигмоиды)
hidden_activation = 1 / (1 + np.exp(-hidden_layer_values))

# Пересчёт значений выходного слоя
output = np.dot(hidden_activation, output_weights)
print("Результат:", output)
