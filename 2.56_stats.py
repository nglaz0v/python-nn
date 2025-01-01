"""
Пример информационной энтропии для расчёта энтропии в узле дерева решений.
"""

import numpy as np

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = np.sum(probabilities * np.log2(probabilities))
    return entropy

# Пример меток классов
labels = [0, 1, 0, 1, 1]

# Рассчитываем информационную энтропию для данного набора меток классов
entropy_value = entropy(labels)
print("Information entropy:", entropy_value)
