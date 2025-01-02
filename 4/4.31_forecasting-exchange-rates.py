"""
Пример на Python с использованием библиотек Pandas, Matplotlib и Sklearn для
анализа макроэкономических данных, финансовых показателей, новостей и других
факторов, влияющих на курс валют (для упрощения примера будем рассматривать
одну валюту).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Загрузка данных, например, курса валют и факторов, которые могут на него влиять (например, ставки, индексы, новости)
# Предположим, что данные хранятся в CSV файлах, которые затем загружаются с использованием pandas
exchange_rate_data = pd.read_csv('exchange_rate.csv')
economic_data = pd.read_csv('economic_indicators.csv')
news_data = pd.read_csv('news_sentiment.csv')

# Объединение данных в один DataFrame
merged_data = pd.merge(exchange_rate_data, economic_data, on='date')
merged_data = pd.merge(merged_data, news_data, on='date')

# Предварительная обработка данных, например, заполнение пропущенных значений, шкалирование при необходимости

# Разделение данных на обучающую и тестовую выборки
X = merged_data.drop(['exchange_rate'], axis=1)  # Входные признаки
y = merged_data['exchange_rate']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание курса валюты для тестовой выборки
y_pred = model.predict(X_test)

# Визуализация результатов
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Exchange Rate')
plt.ylabel('Predicted Exchange Rate')
plt.title('Actual vs Predicted Exchange Rate')
plt.show()
