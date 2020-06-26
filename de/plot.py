import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv
import itertools


def get_data(path, pop=None, n=None, cr=None, f=None):
    def eq_or_none(a, b):
        return a == b or b is None

    x = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            row_success, row_evaluations, row_cr, row_f, row_n, row_pop, row_distance = [float(x) for x in row]
            if eq_or_none(row_pop, pop) and eq_or_none(row_n, n) and eq_or_none(row_cr, cr) and eq_or_none(row_f, f):
                x.append([row_success, row_evaluations, row_cr, row_f, row_n, row_pop, row_distance])
    return np.array(x)


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_parameters(file, xkey, ykey, zkey, k1key, k2key, xonly=None, yonly=None, zonly=None, k1only=None, k2only=None):
    raw_data = get_data(file)
    values = {
        "successrate": raw_data[:, 0],
        "eval": raw_data[:, 1].astype("int"),
        "cr": raw_data[:, 2],
        "f": raw_data[:, 3],
        "n": raw_data[:, 4].astype("int"),
        "pop": raw_data[:, 5].astype("int"),
        "distance": raw_data[:, 6]
    }

    longname = {
        "successrate": "Success rate",
        "eval": "Evaluations on completion",
        "cr": "CR",
        "f": "F",
        "n": "Number of circles",
        "pop": "Population size",
        "distance": "Average distance to optimal"
    }

    xs = values[xkey]
    ys = values[ykey]
    zs = values[zkey]
    k1s = values[k1key]
    k2s = values[k2key]

    f_k1s = set(val for val in k1s if k1only is None or val in k1only)
    f_k2s = set(val for val in k2s if k2only is None or val in k2only)

    perms = list(itertools.product(f_k1s, f_k2s))
    perms = sorted(perms, key=lambda x: (x[0], x[1]))

    for k1, k2 in perms:
        k1_idxs = np.where(k1s == k1)
        k2_idxs = np.where(k2s == k2)
        kx_idxs = np.intersect1d(k1_idxs, k2_idxs)
        handles = []
        for i, z in enumerate(np.unique([val for val in zs[kx_idxs] if zonly is None or val in zonly])):
            colors = get_cmap(len(np.unique(zs[kx_idxs])) + 1)
            z_idxs = np.intersect1d(np.where(zs == z), kx_idxs)
            zstr = str(z) if type(z) == np.int64 else "{:.3f}".format(z)
            handles.append(Patch(color=colors(i), label=f"{zkey}=" + zstr))
            x_idxs = [i for i, val in enumerate(xs) if xonly is None or val in xonly]
            y_idxs = [i for i, val in enumerate(ys) if yonly is None or val in yonly]
            xy_idxs = np.intersect1d(x_idxs, y_idxs)
            x = xs[np.intersect1d(xy_idxs, z_idxs)]
            y = ys[np.intersect1d(xy_idxs, z_idxs)]
            plt.plot(x, y, "o-", color=colors(i))

        plt.title(f"{longname[k1key]}={k1}, {longname[k2key]}={k2}")
        plt.xlabel(longname[xkey])
        plt.yscale("log")
        plt.ylabel(longname[ykey])
        plt.ylim([np.min(ys), np.max(ys) * 1.1])
        plt.legend(handles=handles)

        plt.minorticks_off()
        plt.show()
