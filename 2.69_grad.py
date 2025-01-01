"""
Пример шага градиентного спуска
"""

import tensorflow as tf

# Гиперпараметры
learning_rate = 0.01
epochs = 100
batch_size = 32

# Загрузка данных
mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = x_train / 255.0, tf.keras.utils.to_categorical(y_train, 10)

# Создание модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели с оптимизаторам градиентного спуска SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели с использованием градиентного спуска
for epoch in range(epochs):
    for batch in range(len(x_train) // batch_size):
        start = batch * batch_size
        end = start + batch_size
        x_batch, y_batch = x_train[start:end], y_train[start:end]

        with tf.GradientTape() as tape:
            predictions = model(x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

print("Training finished!")
