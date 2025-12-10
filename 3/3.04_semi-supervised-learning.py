"""
Пример полуобученного обучения с использованием библиотеки scikit-learn 
на Python.
"""

import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation


label_prop_model = LabelPropagation()
iris = datasets.load_iris()
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels = np.copy(iris.target)
labels[random_unlabeled_points] = -1
result = label_prop_model.fit(iris.data, labels)
labels_pred = result.predict(iris.data)

print(labels)
print(labels_pred)
