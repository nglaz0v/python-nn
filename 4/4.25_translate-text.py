"""
Пример кода на Python, использующий библиотеку Googletrans для перевода текста
с одного языка на другой.
"""

from googletrans import Translator

# Создание объекта переводчика
translator = Translator()

# Текст для перевода
text = "Привет, как дела?"

# Определение исходного языка текста
detected_lang = translator.detect(text).lang

# Перевод текста на английский язык
translated_text = translator.translate(text, src=detected_lang, dest='en')

# Вывод переведенного текста
print("Переведённый текст на английский:", translated_text.text)
