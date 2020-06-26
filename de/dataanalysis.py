from de.plot import get_data
import numpy as np
from itertools import product

#for cr in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    #data = get_data("out/alt_mulig_rart4xxx.csv", cr=cr)
    #print(f"CR={cr},\tmeandist={np.mean(data[:, 6], axis=0)}")


def f_stuff(cr=None, pop=None):
    fs = [0.1, 0.48, 0.86, 1.24, 1.62, 2]
    f_means = []
    for f in fs:
        data = get_data("out/alt_mulig_rart4xxx.csv", pop=pop, cr=cr, f=f)
        f_mean = np.mean(data[:, 6], axis=0)
        f_means.append(f_mean)
        print(f"F={f},\tmeandist={f_mean}")

    best_f_idx = int(np.argmin(f_means))
    print(f"Best F={fs[best_f_idx]}")


def pop_stuff(cr=None, f=None):
    pops = [20, 40, 60, 80, 100, 120]
    pop_means = []
    for pop in pops:
        data = get_data("out/alt_mulig_rart4xxx.csv", pop=pop, cr=cr, f=f)
        pop_mean = np.mean(data[:, 6], axis=0)
        pop_means.append(pop_mean)
        print(f"pop={pop},\tmeandist={pop_mean}")
    best_f_idx = int(np.argmin(pop_means))
    print(f"Best pop={pops[best_f_idx]}")

# pop_stuff(0.8)


for n in np.arange(6, 17, 2):
    data = get_data("out/alt_mulig_rart4xxx.csv", cr=0, n=n)
    best_row = np.argmin(data[:, 6])
    print(f"n={n},\tCR={data[best_row, 2]},\tF={data[best_row, 3]},\tpop={data[best_row, 5]}, dist={'{:.5f}'.format(data[best_row, 6])}")


def get_pareto_idxs(a):
    idx = []
    for i, row_a in enumerate(a):
        for j, row_b in enumerate(a):
            if i == j:
                continue
            if np.all((row_a - row_b) > 0):  # row_b dominates row_a (minimization)
                break
        else:
            idx.append(i)
    return idx


def all_combinations(use_cached=True):
    datafile = "out/alt_mulig_rart4xxx.csv"
    cachefile = "temp/cache.npy"

    crs = [0.8]
    fs = [0.1, 0.48, 0.86, 1.24, 1.62, 2]
    pops = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    perms = list(product(crs, fs, pops))
    ns = [6, 8, 10, 12, 14, 16]
    if use_cached:
        results = np.load(cachefile)
    else:
        results = np.zeros((len(perms), len(ns)))
        for i, (cr, f, pop) in enumerate(perms):
            for j, n in enumerate(ns):
                score = get_data(datafile, cr=cr, f=f, pop=pop, n=n)[0, 6]
                results[i, j] = score
        np.save(cachefile, results)
    best_at_n = np.argmin(results, axis=0)
    #for idx in best_at_n:
    #    print(perms[idx])
    # pareto_idxs = get_pareto_idxs(results)
    # means = np.mean(results, axis=1)
    # for idx in pareto_idxs:
        # print(f"CR={perms[idx][0]}, F={perms[idx][1]}, POP={perms[idx][2]} -> " + "{:.4f}".format(means[idx]))
        # print(f"n=16 score -> {results[idx, 5]}")


all_combinations(use_cached=True)
