"""Пример детерминанта матрицы."""

def determinant(matrix):
    return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

# Пример матрицы 2х2
matrix = [[2, 3],
          [1, 4]]

print("Детерминант матрицы:", determinant(matrix))
