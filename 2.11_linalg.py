"""Пример скалярного произведения двух векторов."""

def dot_product(vector1, vector2):
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    return result

# Пример векторов
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

print("Скалярное произведение векторов:", dot_product(vector1, vector2))
