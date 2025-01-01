"""
Пример на Python, демонстрирующий обучение ансамблем моделей машинного обучения
с помощью метода бэггинга (bootstrap aggregating) с использованием библиотеки
scikit-learn.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загрузка датасета Iris
iris = load_iris()
X = iris.data
у = iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, у, test_size=0.2, random_state=42)

# Определение базовой модели (в данном случае решающего дерева)
base_model = DecisionTreeClassifier()

# Определение ансамбля моделей с помощью бэггинга
bagging_model = BaggingClassifier(base_model, n_estimators=10, random_state=42)

# Обучение ансамбля
bagging_model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = bagging_model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Bagging Ensemble:", accuracy)
