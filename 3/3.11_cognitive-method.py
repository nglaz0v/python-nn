"""
Пример использования библиотеки TensorFlow для создания глубокой нейронной сети
с использованием набора данных MNIST.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка данных (пример МNIST)
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация данных
train_images = train_images / 255.0
test_images = test_images / 255.0

# Определение модели нейронной сети
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=5)

# Оценка точности модели на тестовом наборе данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)
