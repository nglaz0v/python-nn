"""
Пример вычисления производной функции потерь.
"""

# Пример вычисления производной функции потерь по параметру веса в нейронной сети

# Гипотетическая функция потерь
def loss_function(y_true, y_pred):
    return (y_true - y_pred) ** 2

# Пример вычисления производной функции потерь по параметру веса
def compute_weight_gradient(input_value, true_output, predicted_output, weight):
    # Вычисляем градиент функции потерь по параметру веса
    loss_gradient = 2 * (true_output - predicted_output) * input_value
    return loss_gradient

# Пример использования производной для обновления веса в нейронной сети
learning_rate = 0.01
true_output = 1
predicted_output = 0.8
weight = 0.5
input_value = 0.6

# Вычисляем градиент функции потерь по параметру веса
gradient = compute_weight_gradient(input_value, true_output, predicted_output, weight)

# Обновляем вес с использованием градиента и скорости обучения
updated_weight = weight - learning_rate * gradient
print("Updated weight:", updated_weight)
