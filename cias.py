import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def evaluate(x):
    points = as_points(x)
    dists = cdist(points, points)
    return dists[np.triu_indices(dists.shape[0], 1, dists.shape[0])].min()


def as_points(x):
    return np.asarray([[x[i], x[i+1]] for i in range(0, len(x), 2)])


def plot(x):
    points = as_points(x)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
