"""
Пример определения свёрточной нейронной сети с двумя свёрточными слоями,
пулингом и двумя полносвязными слоями на языке программирования Python с
использованием библиотеки TensorFlow.
"""

import tensorflow as tf

# Определение модели
model = tf.keras.models.Sequential([
    # Свёрточный слой 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Пулинг слой 1
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Свёрточный слой 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Пулинг слой 2
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Преобразование многомерного вектора в одномерный
    tf.keras.layers.Flatten(),
    # Полносвязный слой 1
    tf.keras.layers.Dense(128, activation='relu'),
    # Полносвязный слой 2
    tf.keras.layers.Dense(10, activation='softmax')  # 10 классов для классификации
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()
