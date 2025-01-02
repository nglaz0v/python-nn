"""
Пример полуобученного обучения с использованием библиотеки scikit-learn на
Python.
"""

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Некоторые обучающие данные
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 1, 0, 1])

# Некоторые неразмеченные данные
X_unlabeled = np.array([[1.5, 2.5], [3.5, 4.5]])

# Создание модели полуобученного обучения (Stochastic Gradient Descent Classifier)
model = make_pipeline(StandardScaler(),
                      SGDClassifier(loss='log_loss', max_iter=1000))

# Обучение модели на части размеченных данных
model.fit(X_train, y_train)

# Прогнозирование меток классов для неразмеченных данных
predicted_labels = model.predict(X_unlabeled)
print("Predicted labels for unlabeled data:", predicted_labels)
