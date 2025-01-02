"""
Пример маргинального распределения вероятности.
"""

import torch
import torch.nn as nn
import numpy as np

# Создание нейронной сети
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Полносвязный слой с 10 входами и 5 выходами
        self.fc2 = nn.Linear(5, 1)  # Полносвязный слой с 5 входами и 1 выходом

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применение ReLU к выходу первого слоя
        x = self.fc2(x)  # Выходной слой без функции активации
        return x

# Создание экземпляра модели
model = NeuralNetwork()
# Генерация случайных входных данных (например, признаков)
input_data = torch.randn(100, 10)  # Генерация 100 случайных примеров с 10 признаками каждый
# Прямой проход
outputs = model(input_data)
# Вычисление маргинального распределения вероятности
marginal_distribution = torch.sigmoid(outputs)  # Применение сигмоидной функции для получения вероятности

# Вывод результатов
print(marginal_distribution)
