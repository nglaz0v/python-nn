"""
Пример гибридного обучения, включающего в себя комбинацию нейронной сети и
метода опорных векторов (Support Vector Machine, SVM) на языке Python с
использованием библиотеки scikit-learn.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Загрузка датасета Iris
iris = load_iris()
X = iris.data
у = iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, у, test_size=0.2, random_state=42)

# Определение моделей для гибридного обучения
svm_model = SVC(kernel='linear', probability=True)
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Объединение моделей в голосующий классификатор
hybrid_model = VotingClassifier(estimators=[('svm', svm_model),
                                            ('nn', nn_model)], voting='soft')

# Обучение гибридной модели
hybrid_model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = hybrid_model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Hybrid Model:", accuracy)
