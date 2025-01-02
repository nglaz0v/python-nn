"""
Пример использования распределения вероятности.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Создание нейронной сети
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Полносвязный слой с 784 входами и 128 выходами
        self.fc2 = nn.Linear(128, 10)  # Полносвязный слой с 128 входами и 10 выходами

    def forward(self, x):
        x = torch.flatten(x, 1)  # Преобразование входного изображения в одномерный тензор
        x = torch.relu(self.fc1(x))  # Применение ReLU к выходу первого слоя
        x = self.fc2(x)  # Выходной слой без функции активации (после него будет применена softmax)
        return x

# Создание экземпляра модели
model = NeuralNetwork()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()  # Функция потерь Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Стохастический градиентный спуск

# Генерация случайных входных данных (например, изображений)
input_data = torch.randn(10, 28, 28) # Генерация 10 случайных изображений размером 28х28

# Прямой проход
outputs = model(input_data)

# Вычисление потерь
labels = torch.randint(0, 10, (10,)) # Случайные метки классов для примеров
loss = criterion(outputs, labels)

# Обратное распространение ошибки и обновление весов
optimizer.zero_grad()
loss.backward()
optimizer.step()
