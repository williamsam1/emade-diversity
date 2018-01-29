from Main import *

from deap import base
from deap import creator
from deap import tools
from math import exp

import random

from AnnealedNSGA import random_merge_sort


def list_dist(a, b, key=None):
    if key is None:
        key = lambda x: x
    return sum([abs(key(a[i]) - key(b[i])) for i in range(len(a))])

"""
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    return [sum(individual)]

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("vary", varOr)
toolbox.register("select", nsga2_selection)


def main():
    pop = toolbox.population(n=300)
    hof = tools.ParetoFront()
    evolve(pop, toolbox, 300, 300, 0.5, 0.2, 50, hall_of_fame=hof)
    for x in hof.items:
        print(x.fitness)

main()
"""
