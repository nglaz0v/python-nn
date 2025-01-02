"""
Пример реализации алгоритма К-средних для сегментации изображения на Python с
использованием библиотеки OpenCV.
"""

import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('image.jpg')

# Преобразование изображения в двумерный массив точек (пикселей)
pixel_values = image.reshape((-1, 3))

# Применение алгоритма К-средних
k = 3  # количество кластеров (сегментов)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values.astype(np.float32), k, None,
                                criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Преобразование центров кластеров в целочисленный формат
centers = np.uint8(centers)

# Присвоение каждому пикселю его центра кластера
segmented_image = centers[labels.flatten()]

# Преобразование обратно в форму изображения
segmented_image = segmented_image.reshape(image.shape)

#Вывод и сохранение сегментированного изображения
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('segmented_image.jpg', segmented_image)
