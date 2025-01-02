"""
Пример уравнения загрузки набора данных Iris.
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Загрузка датасета Iris
iris = load_iris()
X = iris.data
y = iris.target

# Создание и обучение модели дерева решений
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

# Вывод структуры дерева решений в виде текста
tree_rules = export_text(clf, feature_names=iris.feature_names)
print(tree_rules)
