import numpy as np


def de(fobj, bounds, f=0.8, cr=0.7, popsize=20, maxevals=300000, optimal=None, eps=1e-4, r=np.random.RandomState()):
    dimensions = len(bounds)
    pop = r.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    evals = popsize
    while abs(fitness[best_idx] - optimal) > eps and evals < maxevals:
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[r.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + f * (b - c), 0, 1)
            cross_points = r.rand(dimensions) < cr
            if not np.any(cross_points):
                cross_points[r.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        evals += popsize
        yield {"best": best,
               "fitness": fitness[best_idx],
               "evaluations": evals,
               "success": abs(fitness[best_idx] - optimal) < eps}
