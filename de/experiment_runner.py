import multiprocessing as mp
from de.generator_de import de, Crossover
import numpy as np
from time import time
import cias
import itertools
from de.storage import find_stored
from os.path import exists


def print_status(mean_time, cycles_done, cycles_total):
    status_msg = "{:.3f}% completed. ".format(100 * cycles_done / cycles_total)
    if not np.isnan(mean_time):
        status_msg += "Average time: {:.2f}s. ".format(mean_time)
        hours = int((cycles_total - cycles_done) * mean_time // 3600)
        mins = int(((cycles_total - cycles_done) * mean_time % 3600) / 60)
        time_left = (str(hours) + ' hours and ') * bool(hours) + str(mins) + ' minutes'
        status_msg += "Time left: ~" + time_left
    print(status_msg)


def run(n_circles, f, cr, pop, max_evaluations, eps, wb, wb_init, i):
    r = np.random.RandomState(i)
    n_params = n_circles * 2
    generator = de(cias.negative_evaluate,
                   n_params,
                   f=f,
                   cr=cr,
                   popsize=pop,
                   maxevals=max_evaluations,
                   eps=eps,
                   r=r,
                   wb_crossover=wb,
                   wb_init=wb_init)
    result = list(generator)[-1]
    return np.asarray([int(result["success"]), result["evaluations"], result["fitness"] - cias.optimal_scores[n_circles]])


def experiment(pops, ns, fs, crs, reps, eps, wb, wb_init, workingdir="out", filename=f"out-{int(time() // 100)}"):
    while exists(f"{workingdir}/{filename}.csv"):
        filename += "x"

    out = []

    total_cycles = len(pops) * len(ns) * len(fs) * len(crs)
    times = np.zeros(int(total_cycles))
    times[:] = np.nan

    count = 0
    print("{:.3f}% completed".format(0))
    for pop, n, f, cr in itertools.product(pops, ns, fs, crs):
        start_t = time()
        maxeval = n * 10000
        resultrow = find_stored(pop, n, f, cr, path=workingdir)
        if resultrow is None:
            pool = mp.Pool(mp.cpu_count())
            results = np.asarray(pool.starmap(run, [(n, f, cr, pop, maxeval, eps, wb, wb_init, i) for i in range(reps)]))
            pool.close()
            successrate = np.mean(results[:, 0])
            evals = np.mean(results[:, 1])
            dist_to_optimal = np.mean(results[:, 2])
            std = np.std(results[:, 2])
            resultrow = [successrate, np.clip(evals, 0, maxeval), cr, f, n, pop, dist_to_optimal, std]
            times[count] = time() - start_t
        count += 1
        print_status(np.nanmean(times), count, total_cycles)
        out.append(resultrow)
        np.savetxt(f"{workingdir}/{filename}.csv", np.asarray(out), delimiter=",")


# Experiment 1 with grid init
experiment(
    crs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    fs=[0.1, 0.48, 0.86, 1.24, 1.62, 2],
    pops=[20, 40, 60, 80, 100, 120],
    ns=[6, 8, 10, 12, 14, 16],
    reps=8,
    eps=1e-4,
    workingdir="wb_init",
    filename="experiment-1",
    wb=None,
    wb_init=True
)

# Experiment 1 with weighted wb crossover
#experiment(
#    crs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
#    fs=[0.1, 0.48, 0.86, 1.24, 1.62, 2],
#    pops=[20, 40, 60, 80, 100, 120],
#    ns=[6, 8, 10, 12, 14, 16],
#    reps=8,
#    eps=1e-4,
#    workingdir="wbo_weighted",
#    wb=Crossover.WEIGHTED
#)

# experiment(pops=[90, 100, 110, 120],
#            ns=[23, 25, 36],
#            fs=[0.1, 0.48, 0.86, 1.24, 1.62, 2],
#            crs=[0.75, 0.8, 0.85, 0.9, 0.95, 1],
#            reps=8,
#            eps=1e-4,
#            filename="wbo/finjustering")
