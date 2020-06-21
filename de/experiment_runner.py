import multiprocessing as mp
from de.generator_de import de
import numpy as np
from time import time
import cias
import itertools


def print_status(mean_time, cycles_done, cycles_total):
    status_msg = "{:.3f}% completed. ".format(100 * cycles_done / cycles_total)
    status_msg += "Average time: {:.2f}s. ".format(mean_time)
    hours = int((cycles_total - cycles_done) * mean_time // 3600)
    mins = int(((cycles_total - cycles_done) * mean_time % 3600) / 60)
    time_left = (str(hours) + ' hours and ') * bool(hours) + str(mins) + ' minutes'
    status_msg += "Time left: ~" + time_left
    print(status_msg)


def run(n, f, cr, pop, max_evaluations, eps, i):
    r = np.random.RandomState(i)
    generator = de(cias.negative_evaluate,
                   [(0, 1)] * n,
                   f=f,
                   cr=cr,
                   popsize=pop,
                   maxevals=max_evaluations,
                   optimal=cias.optimal_scores[n / 2],
                   eps=eps,
                   r=r)
    result = list(generator)[-1]
    return np.asarray([int(result["success"]), result["evaluations"], result["fitness"] - cias.optimal_scores[n / 2]])


def experiment(pops, ns, fs, crs, reps, eps, filename=f"out-{int(time() // 100)}"):
    out = []

    total_cycles = len(pops) * len(ns) * len(fs) * len(crs)
    times = np.zeros(int(total_cycles))

    count = 0
    print("{:.3f}% completed".format(0))

    for pop, n, f, cr in itertools.product(pops, ns, fs, crs):
        maxeval = n * 10000
        start_t = time()
        pool = mp.Pool(mp.cpu_count())
        results = np.asarray(pool.starmap(run, [(n, f, cr, pop, maxeval, eps, i) for i in range(reps)]))
        pool.close()
        successrate = np.mean(results[:, 0])
        evals = np.mean(results[:, 1])
        dist_to_optimal = np.mean(results[:, 2])
        out.append([successrate, evals, cr, f, n / 2, pop, dist_to_optimal])
        times[count] = time() - start_t
        count += 1
        print_status(np.mean(times[:count]), count, total_cycles)
        savearr = np.asarray(out)
        savearr[:, 1] = np.clip(savearr[:, 1], 0, maxeval)
        np.savetxt(f"out/{filename}.csv", savearr, delimiter=",")


experiment(pops=[40, 60, 80, 100, 120],
           ns=np.arange(12, 24 + 1, 4),
           fs=np.linspace(0.1, 2, num=7),
           crs=np.linspace(0, 1, num=5),
           reps=8,
           eps=1e-4,
           filename="alt_mulig_rart")
