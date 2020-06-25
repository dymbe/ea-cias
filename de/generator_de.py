import numpy as np
import cias
from enum import Enum


class Crossover(Enum):
    CLOSEST_POINT = 1
    WEIGHTED = 2


def de(fobj, n, f=0.8, cr=0.7, popsize=20, maxevals=300000, eps=1e-4, r=np.random.RandomState(),
       wb_crossover=None):
    pop = r.rand(popsize, n)
    fitness = np.asarray([fobj(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    evals = popsize
    optimal = cias.optimal_scores[n / 2]
    popdistances = np.asarray([cias.get_negative_distances(ind) for ind in pop]) if wb_crossover is not None else None
    while abs(fitness[best_idx] - optimal) > eps and evals < maxevals:
        for i in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = pop[r.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + f * (b - c), 0, 1)

            if wb_crossover == Crossover.CLOSEST_POINT:
                smallest_distances = np.max(popdistances[i], axis=0)
                smallest_idxs = np.where(smallest_distances == fitness[i])[0] * 2
                cross_points = np.zeros(n)
                for idx in smallest_idxs:
                    cross_points[[idx, idx + 1]] = True
            elif wb_crossover == Crossover.WEIGHTED:
                smallest_distances = np.max(popdistances[i], axis=0)
                sorted_idxs = np.argsort(smallest_distances) * 2
                weights = np.zeros(n)
                for j, idx in enumerate(sorted_idxs):
                    x = j / (len(sorted_idxs) - 1)
                    weights[[idx, idx + 1]] = 1 - ((1 - cr) / cr) * x
                cross_points = r.rand(n) < weights
            else:
                cross_points = r.rand(n) < cr
                if not np.any(cross_points):
                    cross_points[r.randint(0, n)] = True

            trial = np.where(cross_points, mutant, pop[i])

            if wb_crossover is not None:
                new_distances = cias.get_negative_distances(trial)
                f = np.max(new_distances)
            else:
                new_distances = None
                f = fobj(trial)
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = trial
                if wb_crossover is not None:
                    popdistances[i] = new_distances
                if f < fitness[best_idx]:
                    best_idx = i
                    best = trial
        evals += popsize
        yield {"best": best,
               "fitness": fitness[best_idx],
               "evaluations": evals,
               "success": abs(fitness[best_idx] - optimal) < eps}
