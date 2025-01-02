"""
Пример на Python, который использует библиотеку SpeechRecognition для
распознавания речи и библиотеку Googletrans для машинного перевода текста.
"""

import speech_recognition as sr
from googletrans import Translator

# Создание объектов Recognizer и Translator
recognizer = sr.Recognizer()
translator = Translator()

# Загрузка аудиофайла
audio_file = "audio.wav"
with sr.AudioFile(audio_file) as source:
    # Запись аудио из файла
    audio_data = recognizer.record(source)

    # Преобразование аудио в текст с использованием Google Web Speech API
    text = recognizer.recognize_google(audio_data, language="ru-RU")

    # Перевод текста на целевой язык
    translated_text = translator.translate(text, dest='en').text

    # Вывод результата
    print("Teкcт на исходном языке:", text)
    print("Переведенный текст:", translated_text)
