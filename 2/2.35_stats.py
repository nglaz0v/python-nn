"""
Пример использования условной вероятности для принятия решений в условиях
неопределённости с целью предсказания.
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

# Генерация случайных входных данных (например, изображений)
input_data = torch.randn(1, 784) # Генерация случайного изображения размером 28х28

# Прямой проход
outputs = model(input_data)

# Применение softmax для получения вероятностей классов
softmax = nn.Softmax(dim=1)
probs = softmax(outputs)

# Выбор класса с наибольшей условной вероятностью
predicted_class = torch.argmax(probs, dim=1).item()

# Вывод результата предсказания
print("Предсказанный класс:", predicted_class)
