from numpy.random import choice

from Evolve import *


def trivial_removing_classifier_selection(individuals, k, cutoff_fp=0.9, cutoff_fn=0.9):
    """
        Selection method
        Similar to NSGA, but removes all classifiers that are above a
        specific cutoff false positive or false negative rate in order
        to bias against trivial solutions
    """
    nontrivial = remove_trivial(individuals, k, cutoff_fp, cutoff_fn)
    pareto_fronts = sort_nondominated(nontrivial, k)

    for front in pareto_fronts:
        assign_crowding_distance(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def remove_trivial(individuals, k, cutoff_fp=0.9, cutoff_fn=0.9):
    """
        Removes all classifiers with false positive or false negative
        rate above the given cutoffs, leaving k individuals
    """
    if k <= len(individuals):
        return individuals
    nontrivial = [x for x in individuals if x.fitness[0] < cutoff_fp and x.fitness[1] < cutoff_fn]
    trivial = [x for x in individuals if x.fitness[0] >= cutoff_fp or x.fitness[1] > cutoff_fn]

    if k < len(nontrivial):
        return nontrivial
    else:
        return nontrivial + choice(trivial, k - len(nontrivial), False)