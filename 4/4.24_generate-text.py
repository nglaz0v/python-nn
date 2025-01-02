"""
Пример кода на Python, который создаёт текст на основе заданных параметров или
контекста.
"""

import random

# Список слов для генерации текста
nouns = ["кот", "собака", "дом", "автомобиль", "парк", "река"]
adjectives = ["большой", "маленький", "красивый", "старый", "новый", "зелёный"]
verbs = ["бежит", "прыгает", "плавает", "летит", "идёт", "сидит"]
adverbs = ["быстро", "медленно", "активно", "тихо", "громко", "весело"]

# Генерация случайного предложения
def generate_sentence():
    sentence = f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} {random.choice(adverbs)}."
    return sentence.capitalize()

# Генерация текста на основе заданного количества предложений
def generate_text(num_sentences):
    text = ""
    for _ in range(num_sentences):
        text += generate_sentence() + " "
    return text

# Генерация текста из 5 предложений
generated_text = generate_text(5)
print(generated_text)
