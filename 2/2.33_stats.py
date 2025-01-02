"""
Пример зависимости переменных условной вероятности на языке программирования
Python с использованием библиотеки PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np

# Создание нейронной сети
class ConditionalProbabilityNN(nn.Module):

    def __init__(self):
        super(ConditionalProbabilityNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Полносвязный слой с 2 входами и 10 выходами
        self.fc2 = nn.Linear(10, 1)  # Полносвязный слой с 10 входами и 1 выходом

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Применение ReLU к выходу первого слоя
        x = torch.sigmoid(self.fc2(x))  # Применение сигмоидной функции к выходу второго слоя
        return x

# Создание экземпляра модели
model = ConditionalProbabilityNN()

# Генерация случайных входных данных (например, признаков)
input_data = torch.randn(100, 2)

# Генерация 100 случайных примеров с 2 признаками каждый

# Прямой проход
outputs = model(input_data)

# Вывод результатов
print(outputs)
