from Evolve import *


def trivial_avoidant_classifier_selection(individuals, k, trivial_distance_func=lambda x: abs(1 - x.fitness[0]) + abs(1 - x.fitness[1])):
    """
        Selection method
        Similar to NSGA, but weights classifiers closer to the trivial
        solutions less favorably in the crowding distance sorting in order
        to bias against trivial solutions
    """
    pareto_fronts = sort_nondominated(individuals, k)

    for front in pareto_fronts:
        assign_trivial_avoidant_crowding_distance(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def assign_trivial_avoidant_crowding_distance(individuals, trivial_distance_func=lambda fp, fn: abs(1 - fp) + abs(1 - fn)):
    """
        Assigns each of the individuals crowding_distance to its crowding
        distance divided by the value of its distance to the trivial functions,
        as given by the trivial_distance_func, in order to bias the crowding
        distance to favor classifiers that are nontrivial
    """
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
        individuals[i].fitness.crowding_dist = dist / trivial_distance_func(individuals[i].fitness.values[0], individuals[i].fitness.values[1])
