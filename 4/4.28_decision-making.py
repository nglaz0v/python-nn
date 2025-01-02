"""
Пример на Python с использованием библиотеки scikit-learn для построения
простой модели машинного обучения, которая может помочь врачу в принятии более
точного решения по диагностике и предсказанию результатов лечения.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных, предположим, у нас есть набор данных о пациентах и их диагнозах
# Х - матрица признаков (характеристики пациентов, например, возраст, пол, результаты анализов)
# у - вектор целевых значений (диагнозы)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, у, test_size=0.2, random_state=42)

# Инициализация и обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание диагнозов для тестовой выборки
y_pred = model.predict(X_test)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Вывод отчета о классификации для более подробной информации о производительности модели
print(classification_report(y_test, y_pred))
