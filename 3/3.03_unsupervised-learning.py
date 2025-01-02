"""
Пример кластеризации данных с использованием алгоритма K-means на Python с
помощью библиотеки scikit-learn.
"""

from sklearn.cluster import KMeans
import numpy as np

# Входные данные (признаки объектов)
Х = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Создание модели кластеризации с 2 кластерами
kmeans = KMeans(n_clusters=2)

# Обучение модели на данных
kmeans.fit(Х)

# Получение меток кластеров для каждого объекта
labels = kmeans.labels_

# Получение координат центров кластеров
centers = kmeans.cluster_centers_

print("Meтки кластеров:", labels)
print("Центры кластеров:", centers)
