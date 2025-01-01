"""
Пример генетического алгоритма для решения задачи оптимизации бинарных строк
на языке Python.
"""

import numpy as np

# Функция приспособленности (fitness function)
def fitness_function(solution):
    target = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])  # Целевой массив
    return np.sum(solution == target)  # Возвращает количество совпадающих элементов

# Инициализация популяции
def initialize_population(pop_size, chromosome_length):
    return np.random.randint(2, size=(pop_size, chromosome_length))

# Отбор особей для скрещивания
def select_parents(population, fitness_values):
    sorted_indices = np.argsort(fitness_values)[::-1]  # Сортируем no убыванию приспособленности
    return population[sorted_indices[:2]]  # Возвращаем двух лучших особей

# Скрешивание (одноточечное)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Мутация
def mutate(child, mutation_rate):
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = 1 - child[i]  # Инвертируем бит
    return child

# Генетический алгоритм
def genetic_algorithm(pop_size, chromosome_length, generations, mutation_rate):
    population = initialize_population(pop_size, chromosome_length)
    for generation in range(generations):
        fitness_values = np.array([fitness_function(individual) for individual in population])
        parents = select_parents(population, fitness_values)
        offspring = []
        for _ in range(pop_size // 2):
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            offspring.append(child1)
            offspring.append(child2)
        population = np.array(offspring)
    best_solution_index = np.argmax(fitness_values)
    best_solution_fitness = fitness_values[best_solution_index]
    best_solution = population[best_solution_index]
    return best_solution, best_solution_fitness

# Параметры генетического алгоритма
pop_size = 20
chromosome_length = 10
generations = 100
mutation_rate = 0.1

# Запуск генетического алгоритма
best_solution, best_fitness = genetic_algorithm(pop_size, chromosome_length,
                                                generations, mutation_rate)
print("Вest solution:", best_solution)
print("Вest fitness: ", best_fitness)
