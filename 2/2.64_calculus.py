"""
Пример уравнения градиентного спуска.
"""

import numpy as np

def f(х):
    """Целевая функция, которую мы хотим минимизировать."""
    return (х - 3) ** 2

def df (х):
    """Производная целевой функции."""
    return 2 * (х - 3)

def gradient_descent(learning_rate, num_iterations):
    """Градиентный спуск для нахождения минимума функции f."""
    # Начальное приближение
    x = np.random.randn()
    history = [x]  # Для хранения значений х на каждой итерации

    for i in range(num_iterations):
        gradient = df(x)
        x -= learning_rate * gradient
        history.append(x)
        print(f"Iteration {i+1}: х = {x}, f(x) = {f(x)}")
    return x, history

# Параметры градиентного спуска
learning_rate = 0.1
num_iterations = 100

optimal_x, history = gradient_descent(learning_rate, num_iterations)
print(f"Optimal х: {optimal_x}")
print(f"Minimum value of f(x): {f(optimal_x)}")

# Визуализация процесса
import matplotlib.pyplot as plt

x_values = np.linspace(-2, 8, 400)
y_values = f(x_values)

plt.plot(x_values, y_values, label='f(x) = (х-З)^2')
plt.scatter(history, [f(x) for x in history], color='red', s=10, label='Gradient Descent Steps')
plt.xlabel('х')
plt.ylabel('f(х)')
plt.legend()
plt.title('Gradient Descent Optimization')
plt.show()
