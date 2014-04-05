__author__ = 'Artiom.Casapu'

import random

population_size = 100
iterations = 100
mutation_prob = 0.15
max_interval = 100
min_interval = -100
cross_prob = 0.4
func = lambda x: x * x
cached_minimum, cached_maximum = None, None

def generate_population():
    return [min_interval + random.random() * (max_interval - min_interval) for _ in xrange(population_size)]

#TODO: think about
def cross_over(x, y):
    return cross_prob * x + (1 - cross_prob) * y

def mutate(x):
    if (random.random() <= mutation_prob):
        if (random.randint(0,9) % 2 == 0):
            sign = 1
        else:
            sign = -1
        return x + sign * (random.random() * (max_interval - min_interval) / 2)
    else:
        return x

def weighted_random_choice(population):
    max = sum(fitness(individual) for individual in population)
    pick = random.uniform(0, max)
    current = 0
    for individual in population:
        current += fitness(individual)
        if current > pick:
            return individual

def drange(starting, ending, step):
     r = starting
     while r < ending:
        yield r
        r += step

def global_maximum():

    global cached_maximum;

    if (cached_maximum != None):
        return cached_maximum

    result = func(min_interval + (max_interval - min_interval) / 2)
    interval = drange(min_interval, max_interval, 0.01)
    for x in interval:
        if (func(x) > result):
            result = func(x)

    cached_maximum = result
    return result

def global_minimum():

    global cached_minimum

    if (cached_minimum != None):
        return cached_minimum

    result = func(min_interval + (max_interval - min_interval) / 2)
    interval = drange(min_interval, max_interval, 0.01)
    for x in interval:
        if (func(x) < result):
            result = func(x)

    cached_minimum = result
    return result

def fitness(x):
    return 1 - abs(func(x)) / (global_maximum() - global_minimum())

def next_population(population):
    new_population = []

    for i in range(len(population)):

        indiv1 = weighted_random_choice(population)
        indiv2 = weighted_random_choice(population)

        new_population.append(cross_over(indiv1, indiv2))

    for i in range(len(new_population)):
        new_population[i] = mutate(new_population[i])

    return new_population

def get_fittest(population):
    return max(population, key=fitness)

def genetic_algorithm():

    population = generate_population()

    for i in range(iterations):
        population = next_population(population)

    return get_fittest(population)

if __name__ == "__main__":
    print genetic_algorithm()