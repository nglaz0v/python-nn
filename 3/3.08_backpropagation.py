"""
Пример метода обратного распространения ошибки на Python.
"""

import numpy as np

# Пример функции активации (сигмоид)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Пример производной функции активации
def sigmoid_derivative(x):
    return x * (1 - x)

# Пример реализации метода обратного распространения ошибки
def backpropagation(input_data, output_data, learning_rate, epochs):
    input_layer_size = input_data.shape[1]

    hidden_layer_size = 4
    output_layer_size = output_data.shape[1]

    # Инициализация весов сети
    weights_input_hidden = np.random.uniform(size=(input_layer_size, hidden_layer_size))
    weights_hidden_output = np.random.uniform(size=(hidden_layer_size, output_layer_size))

    for epoch in range(epochs):
        # Прямое распространение
        hidden_layer_input = np.dot(input_data, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Вычисление ошибки
        output_error = output_data - predicted_output
        if epoch % 10000 == 0:
            print('Error: ', np.mean(np.abs(output_error)))

        # Обратное распространение
        d_predicted_output = output_error * sigmoid_derivative(predicted_output)
        output_layer_error = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer_output = output_layer_error * sigmoid_derivative(hidden_layer_output)

        # Обновление весов
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += input_data.T.dot(d_hidden_layer_output) * learning_rate

    return weights_input_hidden, weights_hidden_output

# Пример использования метода обратного распространения ошибки
input_data = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

output_data = np.array([[0],
                        [1],
                        [1],
                        [0]])

learning_rate = 0.1
epochs = 100000

weights_input_hidden, weights_hidden_output = backpropagation(input_data, output_data, learning_rate, epochs)
print("Weights Input to Hidden:\n", weights_input_hidden)
print("Weights Hidden to Output:\n", weights_hidden_output)
