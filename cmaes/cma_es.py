import cias
import cma
import numpy as np


def default_args():
    fmin_default_args = {
        'sigma0': 0.25
    }
    opts_default_args = {
        'bounds': [0, 1],
        'maxfevals': 10 ** 5
    }
    return fmin_default_args, opts_default_args


def get_optimal_points(n, fmin_args=None, opts_args=None):
    if fmin_args is None:
        fmin_args = default_args()[0]
    if opts_args is None:
        opts_args = default_args()[1]

    opts = cma.CMAOptions(**opts_args)
    res = cma.fmin(cias.negative_evaluate,
                   np.random.rand(n * 2),
                   options=opts,
                   **fmin_args
                   )
    return res[0]


if __name__ == '__main__':
    cias.plot(get_optimal_points(6))
