"""
Пример создания наивного байесовского классификатора.
"""

import numpy as np

class NaiveBayesClassifier:

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_probs = {}
        self.feature_probs = {}

        for c in self.classes:
            X_c = X_train[y_train == c]
            self.class_probs[c] = len(X_c) / len(X_train)
            self.feature_probs[c] = {}
            for i in range(X_train.shape[1]):
                self.feature_probs[c][i] = {}
                for value in np.unique(X_train[:, i]):
                    self.feature_probs[c] [i] [value] = (np.sum(X_c[:, i] == value) + 1) / (len(X_c) + len(np.unique(X_train[:, i])))

    def predict(self, X_test):
        predictions = []
        for x in Х_test:
            probs = {c: np.log(self.class_probs[c]) for c in self.classes}
            for c in self.classes:
                for i, value in enumerate(x):
                    probs[c] += np.log(self.feature_probs[c][i].get(value, 1e-5))  # laplace smoothing
            predictions.append(max(probs, key=probs.get))
        return predictions

# Пример использования
X_train = np.array([[1, 'S'],
                    [1, 'М'],
                    [1, 'М'],
                    [1, 'S'],
                    [1, 'S'],
                    [2, 'S'],
                    [2, 'М'],
                    [2, 'М'],
                    [2, 'L'],
                    [2, 'L'],
                    [3, 'L'],
                    [3, 'М'],
                    [3, 'М'],
                    [3, 'L'],
                    [3, 'L']])
y_train = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])  # 0 - отрицательный класс, 1 - положительный класс

Х_test = np.array([[2, 'S'],
                   [3, 'М']])

nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)
predictions = nb_classifier.predict(Х_test)
print("Predictions:", predictions)
