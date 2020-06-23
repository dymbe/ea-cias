import numpy as np


def de(fobj, n, f=0.8, cr=0.7, popsize=20, maxevals=300000, optimal=None, eps=1e-4, r=np.random.RandomState()):
    pop = r.rand(popsize, n)
    fitness = np.asarray([fobj(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    evals = popsize
    while abs(fitness[best_idx] - optimal) > eps and evals < maxevals:
        for i in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = pop[r.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + f * (b - c), 0, 1)
            cross_points = r.rand(n) < cr
            if not np.any(cross_points):
                cross_points[r.randint(0, n)] = True
            trial = np.where(cross_points, mutant, pop[i])
            f = fobj(trial)
            if f < fitness[i]:
                fitness[i] = f
                pop[i] = trial
                if f < fitness[best_idx]:
                    best_idx = i
                    best = trial
        evals += popsize
        yield {"best": best,
               "fitness": fitness[best_idx],
               "evaluations": evals,
               "success": abs(fitness[best_idx] - optimal) < eps}
