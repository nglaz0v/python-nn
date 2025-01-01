"""Пример уравнения главных компонент."""

from sklearn.decomposition import PCA
import numpy as np

# Пример данных
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Создание объекта РСА и установка количества компонент
pca = PCA(n_components=2)

# Преобразование данных
Х_рса = pca.fit_transform(X)

# Получение главных компонент (собственных векторов)
principal_components = pca.components_
print("Главные компоненты (собственные векторы):")
print(principal_components)

# Получение доли объяснённой дисперсии
explained_variance_ratio = pca.explained_variance_ratio_
print("Дoля объяснённой дисперсии:")
print(explained_variance_ratio)
