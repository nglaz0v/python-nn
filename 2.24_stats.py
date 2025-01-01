"""
Пример расчёта вероятности класса на языке Python в контексте наивного
байесовского классификатора.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Загрузка датасета Iris
iris = load_iris()
X, y = iris.data, iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Создание и обучение модели наивного байесовского классификатора
model = GaussianNB()
model.fit(X_train, y_train)

# Предсказание классов для тестового набора
y_pred = model.predict(X_test)

# Оценка точности классификации
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
