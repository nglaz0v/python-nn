"""
Пример на Python с использованием
библиотеки scikit-learn для построения модели машинного
обучения, которая предсказывает риск развития заболевания
на основе уникальных характеристик пациента
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных, например, набора данных о пациентах с различными характеристиками и историей заболеваний
# Х - матрица признаков (уникальные характеристики пациента, например, возраст, пол, история заболеваний)
# у - вектор целевых значений (риск развития заболевания, например, бинарная переменная: 0 - здоров, 1 - риск)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, у, test_size=0.2, random_state=42)

# Инициализация и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание риска развития заболевания для тестовой выборки
y_pred = model.predict(X_test)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Вывод отчета о классификации для более подробной информации о производительности модели
print(classification_report(y_test, y_pred))
