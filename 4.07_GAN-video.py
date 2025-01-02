# -*- coding: utf-8 -*-
"""
Пример использования генеративно-состязательных сетей для генерации новых
кадров на основе обучающих данных.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Функция для создания генератора GAN
def build_generator(latent_dirn, output_shape):
    model = models.Sequential([
        layers.Dense(128, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(np.prod(output_shape), activation='tanh'),
        layers.Reshape(output_shape)
    ])
    return model

# Гиперпараметры
latent_dim = 100  # Размерность скрытого пространства
image_shape = (64, 64, 3)  # Размеры изображений

# Создание генератора
generator = build_generator(latent_dim, image_shape)

# Генерация новых кадров
num_frames = 100
generated_frames = []

for _ in range(num_frames):
    # Генерация случайного вектора скрытого представления
    latent_vector = np.random.normal(size=(1, latent_dim))

    # Генерация изображения с помощью генератора GAN
    generated_frame = generator.predict(latent_vector)

    # Добавление сгенерированного кадра в список
    generated_frames.append(generated_frame)

import cv2
# Преобразование списка кадров в видео
output_video_path = 'generated_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
width, height = image_shape[1], image_shape[0]
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for frame in generated_frames:
    frame = (frame * 255).astype(np.uint8)  # Преобразование значений пикселей к диапазону [0, 255]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Конвертация из RGB в формат OpenCV
    video_writer.write(frame)

# Закрытие видео-писателя
video_writer.release()

print(f'Video saved as {output_video_path}')
