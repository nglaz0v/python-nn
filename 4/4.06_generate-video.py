"""
Пример создания видео, включающего генерацию изображений для последовательности
кадров на языке Python с использованием библиотеки OpenCV.
"""

import cv2
import numpy as np

# Функция для генерации изображения
def generate_image(width, height):
    # Ваш код генерации изображения
    # Например, создание случайного шума или рисование каких-либо объектов
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return image

# Параметры видео
width = 640
height = 480
fps = 24
duration_sec = 10

# Создание объекта VideoWriter
video_writer = cv2.VideoWriter('generated_video.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Генерация кадров и запись в видео
num_frames = fps * duration_sec
for i in range(num_frames):
    # Генерация изображения для текущего кадра
    frame = generate_image(width, height)

    # Запись кадра в видео
    video_writer.write(frame)

    # Вывод текущего номера кадра
    print(f'Frame {i + 1}/{num_frames} generated')

# Закрытие объекта VideoWriter
video_writer.release()

print('Video generation complete. ')
