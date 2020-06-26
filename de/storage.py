from os import listdir
from os.path import isfile, join, exists

import numpy as np


# Finds a stored result of a run
def find_stored(pop, n, f, cr, path="out"):
    for file in [x for x in listdir(path) if isfile(join(path, x))]:
        data = np.loadtxt(f"{path}/{file}", delimiter=",").reshape(-1, 8)
        for row in data:
            if row[2] == cr and row[3] == f and row[4] == n and row[5] == pop:
                print("Saved result found")
                return [x for x in row]
    return None
