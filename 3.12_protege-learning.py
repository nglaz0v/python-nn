"""
Пример реализации метода обучения по претендентам для решения задачи
оптимизации.
"""

import numpy as np

# Функция, которую мы хотим оптимизировать (пример)
def objective_function(x):
    return -(x ** 2)
# Минимизируем функцию -х^2

# Метод обучения по претендентам (Hill Climbing)
def hill_climbing(objective_function, num_iterations, step_size=0.1):
    # Начальное решение
    current_solution = np.random.uniform(-10, 10)
    # Случайное начальное значение
    current_score = objective_function(current_solution)

    # Поиск оптимального решения
    for _ in range(num_iterations):
        # Генерация нового претендента в окрестности текущего решения
        new_solution = current_solution + np.random.uniform(-step_size, step_size)
        new_score = objective_function(new_solution)

        # Если новое решение лучше, то обновляем текущее
        if new_score > current_score:
            current_solution = new_solution
            current_score = new_score

    return current_solution, current_score

# Параметры метода обучения
num_iterations = 1000
step_size = 0.1

# Запуск метода обучения и вывод результата
best_solution, best_score = hill_climbing(objective_function, num_iterations, step_size)
print("Best solution:", best_solution)
print("Best score:", best_score)
