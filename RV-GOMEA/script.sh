#!/bin/bash

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


#                   - dim - - - tau pop nop dmd srt eva vtr imp tol sec
./RV-GOMEA -f -2 -s 14 32 0 1 0 0.8 180 1 0.9 1 160000 -0.33333 57 0 600
