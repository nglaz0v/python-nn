"""
Пример классификации изображения с использованием сверточной нейронной сети.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# Загрузка датасета (например, МNIST)
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Масштабирование значений пикселей к диапазону [О, l]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Определение архитектуры нейронной сети
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)

# Оценка точности на тестовом наборе данных
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print('Test accuracy:', test_acc)
