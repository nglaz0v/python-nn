"""
Пример на Python, использующий библиотеку gTTS (Google Text-to-Speech) для
преобразования текста в речь.
"""

from gtts import gTTS
import os

# Текст для преобразования
text = "Привет! Я голосовой помощник."

# Создание объекта gTTS с указанием языка (русский) и скорости речи
tts = gTTS(text=text, lang='ru')

# Сохранение аудио в файл
tts.save("output.mp3")

# Воспроизведение аудиофайла
os.system("start output.mp3")
