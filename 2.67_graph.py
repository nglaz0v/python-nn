"""
Пример типа графов нейронные сети.
"""

import tensorflow as tf

# Создание последовательной модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics= ['accuracy'])

# Обучение модели на датасете MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model.fit(x_train, y_train, epochs=5)

# Оценка точности модели на тестовом наборе данных
model.evaluate(x_test, y_test)
