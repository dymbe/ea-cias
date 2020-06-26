import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial.distance import cdist


def evaluate(x):
    points = as_points(x)
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def get_distances(x):
    points = as_points(x)
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    return dists


def negative_evaluate(x):
    return -evaluate(x)


def get_negative_distances(x):
    return -get_distances(x)


def as_points(x):
    return np.asarray(x).reshape(-1, 2)


def plot(x):
    points = as_points(x)
    _, axes = plt.subplots()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.scatter(points[:, 0], points[:, 1])
    score = evaluate(x)
    plt.title("Fitness: {:10.4f}".format(score))
    axes.set_aspect(1)
    for x, y in points:
        draw_circle = plt.Circle((x, y), score / 2, fill=False)
        axes.add_artist(draw_circle)
    plt.show()


def get_optimal_score():
    optimal_distances = {}
    with open("../optimal_distances.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            optimal_distances[int(row[0])] = -float(row[1])
    return optimal_distances


optimal_scores = get_optimal_score()
