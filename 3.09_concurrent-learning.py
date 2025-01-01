"""
Пример простого алгоритма конкурентного обучения на Python, который
демонстрирует конкурентное обучение двух агентов в простой среде с двумя
возможными действиями и двумя возможными состояниями.
"""

import numpy as np

# Класс для агента
class Agent:

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)

    def select_action(self):
        return np.argmax(self.q_values)

    def update_q_values(self, action, reward, learning_rate):
        self.q_values[action] += learning_rate * reward

# Функция среды
def environrnent(agent1, agent2, num_episodes, max_steps, learning_rate):
    for episode in range(num_episodes):
        state = np.random.randint(0, 2)  # Генерация случайного начального состояния
        for step in range(max_steps):
            # Выбор действия от обоих агентов
            action1 = agent1.select_action()
            action2 = agent2.select_action()

            # Награда за выбор действий
            if state == 0: # Если состояние 0
                if action1 == 0 and action2 == 0:  # Если оба агента выбирают действие 0
                    reward1, reward2 = 1, 1
                else:
                    reward1, reward2 = 0, 0
            else:  # Если состояние 1
                if action1 == 1 and action2 == 1:  # Если оба агента выбирают действие 1
                    reward1, reward2 = 1, 1
                else:
                    reward1, reward2 = 0, 0

            # Обновление Q-значений агентов
            agent1.update_q_values(action1, reward1, learning_rate)
            agent2.update_q_values(action2, reward2, learning_rate)

            # Изменение состояния
            state = 1 - state
            # Переход к следующему состоянию

# Создание двух агентов
agent1 = Agent(num_actions=2)
agent2 = Agent(num_actions=2)

# Настройка параметров обучения
num_episodes = 1000
max_steps = 100
learning_rate = 0.1

# Запуск среды и обучение агентов
environrnent(agent1, agent2, num_episodes, max_steps, learning_rate)

# Вывод Q-значений обученных агентов
print("Q-values of Agent 1:", agent1.q_values)
print("Q-values of Agent 2:", agent2.q_values)
