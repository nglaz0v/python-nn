"""Уравнение линейной регрессии с использованием библиотеки scikit-learn."""
from sklearn.linear_model import LinearRegression

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели на обучающих данных
model.fit(X_train, y_train)

# Предсказание значений для тестовых данных
predictions = model.predict(X_test)

# Вывод предсказанных значений для первых 10 объектов тестовых данных
print(predictions[:10])
