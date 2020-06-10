import cias
import cma


def get_optimal_points(n, sigma0=0.25, fevalmax=10 ** 5, seed=None):
    '''
    Get optimal points using CMA-ES
    :param n: number of circles
    :param sigma0: initial standard deviation (default is 1/4 domain width)
    :param fevalmax: max evals (default 10^5)
    :param seed: seed to use for reproducibility
    :return: points found with best fitness
    '''
    opts = cma.CMAOptions()
    opts.set("bounds", [0.0, 1.0])
    opts.set("maxfevals", fevalmax)
    if seed is not None:
        opts.set("seed", seed)
    res = cma.fmin(cias.negative_evaluate, [0, 1] * n, sigma0, opts)
    return res[0]


if __name__ == '__main__':
    cias.plot(get_optimal_points(5, 0.25, seed=43), True)
