"""
Пример кода на Python, который использует библиотеку nltk для анализа текста и
модель на основе слов.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Загрузка стоп-слов
nltk.download('punkt')
nltk.download('stopwords')

# Текст для анализа
text = "Вот пример текста для анализа. Мы будем использовать его для подсчёта количества вхождений каждого слова."

# Токенизация текста
tokens = word_tokenize(text)

# Удаление стоп-слов
stop_words = set(stopwords.words('russian'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Подсчёт количества вхождений каждого слова
word_count = Counter(filtered_tokens)

# Вывод результатов
print("Количество вхождений каждого слова:")
print(word_count)
