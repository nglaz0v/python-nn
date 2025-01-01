"""
Пример обучения модели активным образом (Active Learning) на языке Python с
использованием библиотеки scikit-learn.
"""

from sklearn.datasets import make_classification
from sklearn.ensemЬle import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Генерация синтетиче·ских данных
X, у = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, у, test_size=0.2, random_state=42)

# Инициализация модели
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение модели на первоначальной части данных
initial_sample_size = 100
X_initial, X_remaining, y_initial, y_remaining = train_test_split(X_train, y_train, train_size=initial_sample_size, random_state=42)
model.fit(X_initial, y_initial)

# Оценка качества модели на тестовой выборке
y_pred = model.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)
print("Initial accuracy:", initial_accuracy)

# Активное обучение
query_size = 50
num_queries = 10
for _ in range(num_queries):
    # Выбор примеров для разметки, используя стратегию, например, uncertainty sampling
    uncertainty_scores = model.predict_proba(X_remaining)[:, 0]  # Неопределённость определяется как вероятность принадлежности к классу 0
    query_indices = np.argsort(uncertainty_scores)[:query_size]

    # Обновление обучающей выборки
    X_query = X_remaining[query_indices]
    y_query = y_remaining[query_indices]
    Х_initial = np.vstack([X_initial, X_query])
    y_initial = np.hstack([y_initial, y_query])

    # Удаление выбранных примеров из оставшейся выборки
    X_remaining = np.delete(X_remaining, query_indices, axis=0)
    y_remaining = np.delete(y_remaining, query_indices)

    # Переобучение модели на расширенной обучающей выборке
    model.fit(X_initial, y_initial)

    # Оценка качества модели на тестовой выборке
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy after {} queries: {:.Зf}".format((_ + 1) * query_size, accuracy))
