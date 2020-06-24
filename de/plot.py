import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv
import itertools


def get_data(path, pop=None, n=None, cr=None):
    def eq_or_none(a, b):
        return a == b or b is None

    x = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            row_success, row_evaluations, row_cr, row_f, row_n, row_pop, row_distance = [float(x) for x in row]
            if eq_or_none(row_pop, pop) and eq_or_none(row_n, n) and eq_or_none(row_cr, cr):
                x.append([row_success, row_evaluations, row_cr, row_f, row_n, row_pop, row_distance])
    return np.array(x)


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plotdata():
    pops = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    ns = [12]
    crs = [0.8]
    perms = itertools.product(pops, ns)

    for pop, n in perms:
        data = get_data("out/out-1592522819.038235.csv", pop=pop, n=n / 2)
        cr_data = {}
        for row in data:
            success, evaluations, cr, f, _, _ = row
            if cr in crs:
                if cr in cr_data:
                    cr_data[cr] = np.vstack((cr_data[cr], [success, evaluations, cr, f, n, pop]))
                else:
                    cr_data[cr] = np.array([success, evaluations, cr, f, n, pop])

        handles = []
        colors = ["blue", "red", "green", "purple", "orange", "brown", "teal", "gray"]

        for i, cr in enumerate(cr_data):
            handles.append(Patch(color=colors[i], label="CR={:.3f}".format(cr)))
            plt.plot(cr_data[cr][:, 3], cr_data[cr][:, 0], "o-", color=colors[i])

        plt.title(f"Pop={pop}, n={int(n / 2)}")
        plt.xlabel("F")
        plt.ylabel("Number of evaluations")
        plt.ylim([0, 1.1])
        plt.legend(handles=handles)

        plt.minorticks_off()
        plt.show()


def plot_nxyz(file, xkey, ykey, zkey, k1key, k2key):
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

    colors = ["blue", "red", "green", "purple", "orange", "brown", "teal", "gray"]

    xs = values[xkey]
    ys = values[ykey]
    zs = values[zkey]
    k1s = values[k1key]
    k2s = values[k2key]

    perms = list(itertools.product(set(k1s), set(k2s)))
    perms = sorted(perms, key=lambda x: (x[0], x[1]))

    for k1, k2 in perms:
        k1_idxs = np.where(k1s == k1)
        k2_idxs = np.where(k2s == k2)
        k_both_is = np.intersect1d(k1_idxs, k2_idxs)
        handles = []
        for i, z in enumerate(np.unique(zs[k_both_is])):
            colors = get_cmap(len(np.unique(zs[k_both_is])) + 1)
            z_idxs = np.intersect1d(np.where(zs == z), k_both_is)
            zstr = str(z) if type(z) == np.int64 else "{:.3f}".format(z)
            handles.append(Patch(color=colors(i), label=f"{zkey}=" + zstr))
            plt.plot(xs[z_idxs], ys[z_idxs], "o-", color=colors(i))

        plt.title(f"{k1key}={k1}, {k2key}={k2}")
        plt.xlabel(xkey)
        plt.yscale("log")
        plt.ylabel(ykey)
        plt.ylim([np.min(ys), np.max(ys) * 1.1])
        plt.legend(handles=handles)

        plt.minorticks_off()
        plt.show()


plot_nxyz("out/n_vs_cr.csv", "cr", "distance", "n", "pop", "f")