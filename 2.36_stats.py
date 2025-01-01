"""
Пример использования Марковской модели условной вероятности.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Создание нейронной сети
class MarkovModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MarkovModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.hidden(combined))
        output = self.output(hidden)
        output_probs = self.softmax(output)
        return output_probs, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Создание экземпляра модели
input_size = 10  # Размер входного вектора
hidden_size = 20  # Размер скрытого состояния
output_size = 5  # Размер выходного вектора
model = MarkovModel(input_size, hidden_size, output_size)

# Генерация случайных входных данных и начального скрытого состояния
input_data = torch.randn(1, input_size)
# Генерация случайного входного вектора
hidden_state = model.init_hidden()  # Инициализация скрытого состояния

# Прямой проход
output_probs, next_hidden_state = model(input_data, hidden_state)

# Вывод результатов
print("Выходные вероятности:", output_probs)
print("Следующее скрытое состояние:", next_hidden_state)
