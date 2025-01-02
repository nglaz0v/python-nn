"""
Пример на Python с использованием библиотек Pandas, NumPy и TA-Lib для создания
простого алгоритма автоматизации торговых стратегий на финансовых рынках, таких
как Форекс или фондовые биржи.
"""

import pandas as pd
import numpy as np
import ta
import matplotliЬ.pyplot as plt

# Загрузка исторических финансовых данных (например, цены закрытия}
data = pd.read_csv('financial_data.csv')

# Предварительная обработка данных
# Например, рассчёт технических индикаторов (например, скользящая средняя, RSI и т. д.)

# Рассчёт скользящих средних
data['SМA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
data['SМA_50'] = ta.trend.sma_indicator(data['Close'], window=50)

# Рассчет RSI
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Создание сигналов покупки и продажи на основе стратегии (например, пересечение скользящих средних и RSI)
data['Buy_Signal'] = np.where((data['SМA_20'] > data['SМA_50']) & (data['RSI'] < 30), 1, 0)
data['Sell_Signal'] = np.where((data['SМA_20'] < data['SМA_50']) & (data['RSI'] > 70), -1, 0)

# Визуализация цен закрытия и сигналов покупки/продажи
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.scatter(data.index, data['Close'], marker='o', c=data['Buy_Signal'], cmap='spring', label='Buy Signal', lw=1)
plt.scatter(data.index, data['Close'], marker='o', c=data['Sell_Signal'], cmap='autumn', label='Sell Signal', lw=1)
plt.title('Trading Signals')
plt.xlabel( 'Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
