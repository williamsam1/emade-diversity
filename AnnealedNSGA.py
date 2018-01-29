import random
from math import exp

from collections import defaultdict
from itertools import chain
from operator import attrgetter

from deap import tools


def annealed_nsga(individuals, k, temp):
    """
    NSGA 2 selection using a temperature-based random sort
    """
    pareto_fronts = sort_nondominated(individuals, k)

    for front in pareto_fronts:
        assign_crowding_distance(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = random_merge_sort(pareto_fronts[-1], temp, key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def annealed_nsga_shuffled(individuals, k, temp):
    """
    NSGA 2 selection using a temperature-based random sort
    """
    pareto_fronts = sort_nondominated(individuals, k)

    for front in pareto_fronts:
        assign_crowding_distance(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = random_merge_sort(pareto_fronts[-1], temp, key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])
    random.shuffle(chosen)
    return chosen


def sort_nondominated(individuals, k, first_front_only=False):
    if k == 0:
        return []
    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = map_fit_ind.keys()

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    enum_fits = list(enumerate(fits))

    # Rank first Pareto front
    for i, fit_i in enum_fits:
        for _, fit_j in enum_fits[i + 1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


def assign_crowding_distance(individuals):
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    nobj = len(individuals[0].fitness.values)

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist


def random_merge_sort(l, temp, key=None, reverse=False):
    """
    Sorts the list randomly according to the given temperature
    A temperature of 0 is equivalent to a regular sort of the list
    As the temperature increases the sort tends more towards a random sort
    Performs merge sort where the wrong action is taken with probability
    exp(-temp / abs(key(x) - key(y)))/2 + 1/2, when determining where to
    place objects x and y
    """
    assert temp >= 0, "Temperature must be >= 0"

    def val_key(x):
        if key is None:
            return -x if reverse else x
        else:
            return -key(x) if reverse else key(x)

    if len(l) <= 1:
        return l
    elif len(l) == 2:
        add_0 = False
        r = random.random()
        if val_key(l[0]) == val_key(l[1]):
            add_0 = r <= 0.5
        elif val_key(l[0]) < val_key(l[1]):
            add_0 = r <= exp(-temp / abs(val_key(l[0]) - val_key(l[1])))/2 + 1/2
        else:
            add_0 = r > exp(-temp / abs(val_key(l[0]) - val_key(l[1])))/2 + 1/2
        if add_0:
            return [l[0], l[1]]
        else:
            return [l[1], l[0]]
    else:
        a_len = len(l) // 2
        a = random_merge_sort(l[:a_len], temp, val_key)
        b = random_merge_sort(l[a_len:], temp, val_key)
        s = random_merge(a, b, temp, val_key)
        return s


def random_merge(a, b, temp, key=None, reverse=False):
    """
        Merges the lists randomly according to the given temperature
        A temperature of 0 is equivalent to a regular merge of the list
        As the temperature increases the sort tends more towards a random merge
        Performs a merge where the wrong action is taken with probability
        exp(-temp / abs(key(x) - key(y)))/2 + 1/2, when determining where to
        place objects x and y
    """
    def val_key(x):
        if key is None:
            return -x if reverse else x
        else:
            return -key(x) if reverse else key(x)

    l = []
    i = 0
    j = 0
    while i < len(a) or j < len(b):
        if i == len(a):
            l.append(b[j])
            j += 1
        elif j == len(b):
            l.append(a[i])
            i += 1
        else:
            r = random.random()
            add_a = False
            if val_key(a[i]) == val_key(b[j]):
                add_a = r <= 0.5
            else:
                if key(a[i]) < key(b[j]):
                    add_a = r <= exp(-temp / abs(val_key(a[i]) - val_key(b[j])))/2 + 1/2
                else:
                    add_a = r > exp(-temp / abs(val_key(a[i]) - val_key(b[j])))/2 + 1/2
            if add_a:
                l.append(a[i])
                i += 1
            else:
                l.append(b[j])
                j += 1
    return l


def evolve(initial_population, toolbox, population_size, num_children, cxpb, mutpb, ngen, stats=None, hall_of_fame=None):
    """
    Main evolutionary loop
    """
    population = initial_population
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update hall of fame
    if hall_of_fame is not None:
        hall_of_fame.update(population)

    # Update statistics
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    temp = 1

    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = toolbox.vary(population, toolbox, num_children, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update hall of fame
        if hall_of_fame is not None:
            hall_of_fame.update(offspring)

        # Select the next generation population
        population = toolbox.select(population + offspring, population_size, temp=temp)
        # population = toolbox.select(population + offspring, population_size)
        temp /= 2

        # Update statistics
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    return population, logbook


def varOr(population, toolbox, num_children, cxpb, mutpb):
    """
    Evolutionary step with selection from parents and children
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(num_children):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    return offspring
