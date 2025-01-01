"""
Пример на языке Python, демонстрирующий условную независимость событий при
обучении модели на двух входных признаках.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

# Создание входных данных
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Создание целевых меток
у = np.array([0, 1, 1, 1])

# Логическая операция OR
# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X, у)

# Предсказание на новых данных
new_data = np.array([[0, 0],  # 0 OR 0 = 0
                     [0, 1],  # 0 OR 1 = 1
                     [1, 0],  # 1 OR 0 = 1
                     [1, 1]]) # 1 OR 1 = 1
predictions = model.predict(new_data)

print("Предсказания модели:", predictions)
