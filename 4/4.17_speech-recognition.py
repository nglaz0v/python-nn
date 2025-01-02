"""
Пример на Python, использующий библиотеку SpeechRecognition для преобразования
аудио в текст.
"""

import speech_recognition as sr

# Создание объекта Recognizer
recognizer = sr.Recognizer()

# Загрузка аудиофайла
audio_file = "audio.wav"
with sr.AudioFile(audio_file) as source:
    # Запись аудио из файла
    audio_data = recognizer.record(source)

    # Преобразование аудио в текст с использованием Google Web Speech API
    text = recognizer.recognize_google(audio_data, language="ru-RU")

    # Вывод результата
    print("Teкcт из аудио:", text)
