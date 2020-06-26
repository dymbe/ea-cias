#!/bin/bash
#   -f %d: Sets linkage model that is used. Positive: Use a FOS with elements of %d consecutive variables.
# -1 for full linkage model,
# -2 for dynamic linkage tree learned from the population,
# -3 for fixed linkage tree learned from distance measure,
# -4 for bounded fixed linkage tree learned from distance measure,
# -5 for fixed bounded linkage tree learned from random distance measure.

#  pro: Index of optimization problem to be solved (minimization).
# dim: Number of parameters.
# low: Overall initialization lower bound.
# upp: Overall initialization upper bound.
# rot: The angle by which to rotate the problem.
# tau: Selection percentile (tau in [1/pop,1], truncation selection).
# pop: Population size per normal.
# nop: The number of populations (parallel runs that initially partition the search space).
# dmd: The distribution multiplier decreaser (in (0,1), increaser is always 1/dmd).
# srt: The standard-devation ratio threshold for triggering variance-scaling.
# eva: Maximum number of evaluations allowed.
# vtr: The value to reach. If the objective value of the best feasible solution reaches
#      this value, termination is enforced (if -r is specified).
# imp: Maximum number of subsequent generations without an improvement while the
#      the distribution multiplier is <= 1.0.
# tol: The tolerance level for fitness variance (i.e. minimum fitness variance)
# sec: The time limit in seconds.


#                     - dim - - - tau pop nop dmd srt eva vtr imp tol sec
./RV-GOMEA -f 2 -o -s -b 14 22 0 1 0 0.8 $1 1 0.9 1 110000 -0.39820 209 0 600
