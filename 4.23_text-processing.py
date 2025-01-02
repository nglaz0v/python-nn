"""
Пример кода на Python, использующий библиотеку nltk для классификации текста по
тональности, определения темы текста, классификации спама и идентификации
языка.
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect

# Загрузка стоп-слов и анализатора тональности
nltk.download('punkt')
nltk. download('stopwords')
nltk.download('vader_lexicon')

# Текст для анализа
text = "Это просто удивительное приложение! Я в восторге от его функциональности и удобства. Спасибо разработчикам!"
spam_text = "Получите бесплатный iPhone прямо сейчас! Просто введите свой email и получите его!"

# Анализ тональности
sid = SentimentIntensityAnalyzer()
sentiment_scores = sid.polarity_scores(text)
if sentiment_scores['compound'] >= 0.05:
    sentiment = "Позитивный"
elif sentiment_scores['compound'] <= -0.05:
    sentiment = "Негативный"
else:
    sentiment = "Нейтральный"
print("Тональность текста:", sentiment)

# Определение языка
language = detect(text)
print("Язык текста:", language)

# Классификация спама
spam_keywords = ["бесплатно", "акция", "скидка", "подарок"]
tokens = word_tokenize(spam_text.lower())
spam_count = sum(1 for word in tokens if word in spam_keywords)
if spam_count >= 2:
    is_spam = True
else:
    is_spam = False
print("Этo спам:", is_spam)

# Определение темы текста
topic_keywords = ["приложение", "функциональность", "удобство"]
tokens = word_tokenize(text.lower())
topic_count = sum(1 for word in tokens if word in topic_keywords)
if topic_count >= 2:
    topic = "Приложения"
else:
    topic = "Общее"
print("Тема текста:", topic)
