from deap import base, creator, tools
import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def eval_func(individual):
    # Расчет целевой функции для пары
    return (individual[0],)  # Замените на реальный расчет

toolbox = base.Toolbox()
toolbox.register("pair", np.random.choice, a=final_pairs.index, size=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.pair, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Запуск генетического алгоритма
pop = toolbox.population(n=50)
for gen in range(100):
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    # Селекция и обновление популяции
    pop = toolbox.select(offspring, k=len(pop))