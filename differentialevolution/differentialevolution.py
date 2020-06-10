import numpy as np
import cias


class DifferentialEvolution:
    def __init__(self,
                 n,
                 domains,
                 evaluation_function,
                 population_size=50,
                 f=0.8,
                 cr=0.9,
                 eps=1e-2,
                 monitor_cycle=200,
                 max_iterations=10000):
        self.n = n
        self.domains = domains
        self.evaluation_function = evaluation_function
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.eps = eps
        self.evaluations = 0
        self.population = np.zeros((population_size, n))
        self.scores = self.get_population_scores()
        self.monitor_cycle = monitor_cycle
        self.max_iterations = max_iterations

    def evaluate(self, x):
        self.evaluations += 1
        return self.evaluation_function(x)

    def get_population_scores(self):
        return np.asarray([self.evaluate(x) for x in self.population])

    def make_random_population(self):
        random_population = np.random.random((self.population_size, self.n))
        for i, domain in enumerate(self.domains):
            delta = domain[1] - domain[0]
            offset = domain[0]
            random_population[:, i] = random_population[:, i] * delta + offset
        return random_population

    def evolve(self):
        for i, x in enumerate(self.population):
            ai, bi, ci = np.random.choice(np.delete(np.arange(len(self.population)), i), size=3)
            a, b, c = self.population[[ai, bi, ci]]
            r = np.random.randint(self.n)
            y = np.empty(x.shape)
            for j, domain in enumerate(self.domains):
                if np.random.random() < self.cr or j == r:
                    y[j] = a[j] + self.f * (b[j] - c[j])
                    if y[j] < domain[0]:
                        y[j] = domain[0]
                    if y[j] > domain[1]:
                        y[j] = domain[1]
                else:
                    y[j] = x[j]
            new_score = self.evaluate(y)
            if new_score <= self.scores[i]:
                self.population[i] = y
                self.scores[i] = new_score

    def show_progress(self, count):
        if count % 1000 == 0:
            print(f"Iteration: {count}")
            print(f"Best score: {np.min(self.scores)}")
            cias.plot(self.population[np.argmin(self.scores)])

    def optimize(self):
        self.evaluations = 0
        self.population = self.make_random_population()
        self.scores = self.get_population_scores()
        count = 0
        while True:
            prev_min_score = np.min(self.scores)
            self.evolve()  # Evolve population and update score
            self.show_progress(count)
            count += 1
            if count % self.monitor_cycle == 0 and (prev_min_score - np.min(self.scores)) < self.eps:
                break  # Converged
            rd = (np.mean(self.scores) - np.min(self.scores)) ** 2 / (np.min(self.scores) ** 2 + self.eps)
            if rd < self.eps:
                pass#break  # Converged
            if count >= self.max_iterations:
                break  # Not converged, but stop anyway
        print(f"DONE after {self.evaluations} evaluations!")
        return self.population[np.argmin(self.scores)]


if __name__ == '__main__':
    np.random.seed(42)
    n_circles = 5

    de = DifferentialEvolution(2 * n_circles,
                               [(0, 1)] * 2 * n_circles,
                               cias.negative_evaluate,
                               population_size=20 * n_circles,
                               eps=1e-10)

    cias.plot(de.optimize())
