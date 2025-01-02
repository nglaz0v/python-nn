"""
Пример обучения с подкреплением на Python с использованием библиотеки ОрепАI
Gym.
"""

import numpy as np
import gym

# Создание среды MountainCar-v0
env = gym.make('MountainCar-v0')

# Инициализация Q-таблицы (например, нулями)
Q = {}

# Параметры обучения
alpha = 0.1  # Скорость обучения
gamma = 0.9  # Дисконтирование будущих вознаграждений
epsilon = 0.1  # Эксщхорация vs. эксплуатация

# Количество эпизодов для обучения
episodes = 1000

# Обучение агента
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        # Выбор действия с использованием эпсилон-жадной стратегии
        if state not in Q:
            Q[state] = [0, 0, 0]  # Инициализация Q-значений для нового состояния
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Случайное действие (эксплорация)
        else:
            action = np.argmax(Q[state])  # Лучшее известное действие (эксплуатация)

        # Выполнение действия и получение нового состояния и вознаграждения
        new_state, reward, done, _ = env.step(action)

        # Обновление Q-значения для текуmей пары состояние-действие
        if new_state not in Q:
            Q[new_state] = [0, 0, 0]  # Инициализация Q-значений для нового состояния
        Q[state][action] += alpha * (reward + gamma * max(Q[new_state]) - Q[state][action])

        state = new_state

# Пример использования обученной Q-таблицы для выполнения задачи
state = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])  # Выбор действия на основе Q-значений
    new_state, _, done, _ = env.step(action)
    env.render()
    # Отображение среды
    state = new_state

env.close()
# Закрытие окна среды
