"""
Пример уравнения на языке Python для рекуррентной нейронной сети с
использованием библиотеки PyTorch.
"""

import torch
import torch.nn as nn

# Определение архитектуры рекуррентной нейронной сети
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Инициализация скрытого состояния
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Проход по рекуррентным слоям
        out, _ = self.rnn(x, h0)

        # Применение линейного слоя к выходу последнего временного шага
        out = self.fc(out[:, -1, :])
        return out

# Создание экземпляра модели
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 5
model = RNN(input_size, hidden_size, num_layers, output_size)

# Пример использования модели для предсказания
# Предположим, у нас есть тензор input_data размером (batch_size, seq_length, input_size), где
# batch_size - количество образцов в пакете,
# seq_length - длина последовательности,
# input_size - размерность входных данных
input_data = torch.randn(3, 5, 10)
# Пример входных данных (3 образца, каждый с 5 временными шагами и 10 признаками)
output = model(input_data)
print("Результат предсказания:", output)
