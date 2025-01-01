"""Пример вычисления псевдообратной матрицы Мура-Пенроуза."""

import numpy as np

# Создание произвольной матрицы
A = np.array([[1, 2],
              [3, 4]])

# Сингулярное разложение
U, Sigma, Vt= np.linalg.svd(A)

# Вычисление псевдообратной матрицы
Sigma_plus = np.zeros_like(A.T)
Sigma_plus[:len(Sigma), :len(Sigma)] = np.diag(1/Sigma)
A_plus= np.dot(Vt.T, np.dot(Sigma_plus, U.T))

# Проверка результатов
reconstructed_A= np.dot(A_plus, A)
print("Исходная матрица A:\n", A)
print("Псевдообратная матрица A+:\n", A_plus)
print("Пpoвepкa: А+ * A:\n", reconstructed_A)
