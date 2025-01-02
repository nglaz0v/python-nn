"""
Уравнение оценки параметров с использованием метода максимального
правдоподобия.
"""

import numpy as np
from scipy.optimize import minimize

# Генерируем случайные данные для линейной регрессии
np.random.seed(0)
Х = 2 * np.random.rand(100, 1)
у = 4 + 3 * Х + np.random.randn(100, 1)

# Определяем функцию правдоподобия для линейной регрессии
def likelihood(parameters):
    intercept, slope = parameters
    y_pred = intercept + slope * Х
    error = у - y_pred
    likelihood_values = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (error ** 2))
    return -np.prod(likelihood_values)

# Используем метод максимального правдоподобия для оценки параметров
initial_guess = [0, 0]
# Начальное предположение для параметров
result = minimize(likelihood, initial_guess, method='Nelder-Mead')

# Получаем оцененные параметры
intercept_mle, slope_mle = result.x
print("Oцeнкa параметра intercept:", intercept_mle)
print("Oцeнкa параметра slope:", slope_mle)
