"""
Пример на Python, использующий библиотеку Librosa для анализа аудиосигнала и
извлечения различных характеристик, таких как мел-кепстральные коэффициенты
(MFCC), хроматические признаки и темп.
"""

import librosa
import numpy as np

# Загрузка аудиофайла
audio_file = "audio.wav"
audio_data, sample_rate = librosa.load(audio_file)

# Извлечение мел-кепстральных коэффициентов (MFCC)
mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

# Извлечение хроматических признаков
chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)

# Извлечение темпа
tempo = librosa.beat.tempo(y=audio_data, sr=sample_rate)

# Вывод результатов
print("MFCC shape:", mfccs.shape)
print("Chroma shape:", chroma.shape)
print("Tempo:", tempo)
