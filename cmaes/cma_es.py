import cias
import cma
import numpy as np


def get_optimal_points(n, sigma0=0.25, restarts=3, fevalmax=10 ** 5, seed=None):
    '''
    Get optimal points using CMA-ES
    :param n: number of circles
    :param sigma0: initial standard deviation (default is 1/4 domain width)
    :param restarts: number of BIPOP restarts to use
    :param fevalmax: max evals (default 10^5)
    :param seed: seed to use for reproducibility
    :return: points found with best fitness
    '''
    opts = cma.CMAOptions()
    opts.set("bounds", [0.0, 1.0])
    opts.set("maxfevals", fevalmax)
    if seed is not None:
        opts.set("seed", seed)
    res = cma.fmin(cias.negative_evaluate,
                   get_random_initial_population(n),
                   sigma0,
                   opts,
                   bipop=True,
                   restarts=restarts
                   )
    return res[0]


def get_random_initial_population(n):
    pop = []
    for _ in range(n):
        pop.append(np.random.random())
        pop.append(np.random.random())
    return pop


if __name__ == '__main__':
    cias.plot(get_optimal_points(5), True)
