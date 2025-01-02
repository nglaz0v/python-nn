"""
Пример линейной регрессии на Python с использованием библиотеки scikit-learn.
"""

from sklearn.linear_model import LinearRegression
import numpy as np

# Входные данные (площадь дома)
X = np.array([[100], [150], [200], [250], [300]])

# Выходные данные (цена дома)
у = np.array([250000, 350000, 450000, 550000, 650000])

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели на данных
model.fit(X, у)

# Предсказание цены дома для новых данных
new_house_size = np.array([[180]])
predicted_price = model.predict(new_house_size)
print("Predicted price for а house with size 180 sq.rn.:", predicted_price)
