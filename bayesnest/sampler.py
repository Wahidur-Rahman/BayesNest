from typing import Callable, Tuple, Optional, List
import numpy as np
import corner
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from numpy.linalg import eigh
from scipy.special import logsumexp
import time
from bayesnest.utils import generate_cube  # ensure this is defined


class NestedSampler:
    """
    General-purpose log-space nested sampler for Bayesian inference.

    Parameters
    ----------
    log_likelihood : Callable
        A function that computes the log-likelihood given a parameter vector.
    prior : Callable
        A function that transforms a unit cube sample to the prior space.
    ndim : int
        Number of dimensions (parameters) in the model.
    live_points : int, optional
        Number of live points used in the sampler.
    max_iterations : int, optional
        Maximum number of iterations to run.
    verbose : bool
        Whether to print progress information.
    tolerance : float
        Log-evidence tolerance for convergence.
    sampler : str
        Constrained sampling strategy: 'uniform' or 'ellipsoid'.

    Attributes
    ----------
    log_evidence : float
        The natural log of the Bayesian evidence.
    samples : List[np.ndarray]
        Posterior samples.
    log_weights : List[float]
        Log-weights associated with each sample.
    """
    def __init__(
        self,
        log_likelihood: Callable[[np.ndarray], float],
        prior: Callable[[np.ndarray], np.ndarray],
        ndim: int,
        live_points: int = 5,
        max_iterations: int = 400,
        verbose: bool = False,
        tolerance: float = 0.05,
        sampler: str = "ellipsoid",
    ) -> None:

        assert live_points > 2, "Need at least 2 live_points"
        assert tolerance > 0.0, "Tolerance must be positive"
        assert sampler in ("uniform", "ellipsoid"), "sampler must be 'uniform' or 'ellipsoid'"

        self.log_likelihood = log_likelihood
        self.prior = prior
        self.ndim = ndim
        self.live_points = live_points
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tolerance = tolerance
        self.sampler = sampler

        self.log_evidence: float = np.nan
        self.logX: float = 0.0
        self.samples: List[np.ndarray] = []
        self.log_weights: List[float] = []
        self.log_Z_terms: List[float] = []

        # Verify prior transforms correctly
        unit_cube = generate_cube(ndim)
        prior_cube = prior(unit_cube)
        assert len(prior_cube) == ndim, f"Prior dimension mismatch (got {len(prior_cube)}, expected {ndim})"

    def _logsubexp(self, a: float, b: float) -> float:
        if b >= a:
            raise ValueError("_logsubexp requires b < a")
        return a + np.log1p(-np.exp(b - a))

    def _log_beta_shrinkage(self) -> float:
        t = np.random.beta(self.live_points, 1)
        return np.log(t)

    def _sample_from_unit_ball(self, ndim: int) -> np.ndarray:
        x = np.random.normal(0, 1, ndim)
        x /= np.linalg.norm(x)
        r = np.random.rand() ** (1 / ndim)
        return r * x

    def _multi_ellipsoid_sample(self, live_points: List[np.ndarray], likelihood_threshold: float) -> np.ndarray:
        points = np.array(live_points)
        clustering = DBSCAN(eps=2.0, min_samples=3).fit(points)
        labels = clustering.labels_
        clusters = [points[labels == k] for k in set(labels) if k != -1]

        ellipsoids = []
        volumes = []

        for cluster in clusters:
            if len(cluster) < self.ndim + 1:
                continue
            center = np.mean(cluster, axis=0)
            cov = np.cov(cluster.T)
            scale = 1.2
            eigvals, eigvecs = eigh(cov)
            transform = eigvecs @ np.diag(np.sqrt(eigvals) * scale)
            volume = np.prod(np.sqrt(eigvals) * scale)
            ellipsoids.append((center, transform))
            volumes.append(volume)

        volumes = np.array(volumes)
        probs = volumes / np.sum(volumes)

        while True:
            i = np.random.choice(len(ellipsoids), p=probs)
            center, transform = ellipsoids[i]
            u = self._sample_from_unit_ball(self.ndim)
            point = center + transform @ u
            if self.log_likelihood(point) > likelihood_threshold:
                return point

    def run(self) -> None:
        start = time.time()
        unit_cubes = [generate_cube(self.ndim) for _ in range(self.live_points)]
        prior_cubes = [self.prior(cube) for cube in unit_cubes]
        self.likelihood_live = np.array([self.log_likelihood(cube) for cube in prior_cubes])

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"Iteration: {i}")

            i_min = np.argmin(self.likelihood_live)
            l_min = self.likelihood_live[i_min]
            worst = prior_cubes[i_min]
            logX_new = self.logX + self._log_beta_shrinkage()
            log_dX = self._logsubexp(self.logX, logX_new)

            if i == 0:
                log_weight = l_min + log_dX
            else:
                log_avgL = np.logaddexp(l_min, prev_l_min) - np.log(2)
                log_weight = log_avgL + log_dX
            prev_l_min = l_min

            self.samples.append(worst)
            self.log_weights.append(log_weight)
            self.log_Z_terms.append(log_weight)

            log_L_max = np.max(self.likelihood_live)
            log_remaining_Z = log_L_max + logX_new
            log_Z_so_far = logsumexp(self.log_Z_terms)

            if self.verbose:
                print(f"dZ = {log_remaining_Z - log_Z_so_far:.3f} vs log(tol) = {np.log(self.tolerance):.3f}")

            if log_remaining_Z < log_Z_so_far + np.log(self.tolerance):
                if self.verbose:
                    print(f"Stopping early at iteration {i} due to tolerance.")
                break

            if self.sampler == "uniform":
                while True:
                    w_new = self.prior(generate_cube(self.ndim))
                    if self.log_likelihood(w_new) > l_min:
                        break
            elif self.sampler == "ellipsoid":
                w_new = self._multi_ellipsoid_sample(prior_cubes, l_min)

            prior_cubes[i_min] = w_new
            self.likelihood_live[i_min] = self.log_likelihood(w_new)
            self.logX = logX_new

        self.total_iterations = self.max_iterations
        log_X_final = -self.total_iterations / self.live_points
        log_dX_final = np.log(np.exp(log_X_final) / self.live_points)

        for i in range(self.live_points):
            log_weight = self.likelihood_live[i] + log_dX_final
            self.samples.append(prior_cubes[i])
            self.log_weights.append(log_weight)
            self.log_Z_terms.append(log_weight)

        self.log_evidence = logsumexp(self.log_Z_terms)
        end = time.time()
        print(f"Sampling Finished in {end - start:.3f} seconds")

    def get_posterior_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        log_weights = np.array(self.log_weights)
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        return np.array(self.samples), weights

    def get_log_evidence(self) -> float:
        return self.log_evidence

    def plot_posterior(self, labels: Optional[List[str]] = None, truths: Optional[List[float]] = None) -> None:
        samples, weights = self.get_posterior_samples()
        if labels is None:
            labels = [f"w{i}" for i in range(self.ndim)]
        corner.corner(samples, weights=weights, labels=labels, truths=truths,
                      show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
        plt.show()

    def plot_logZ_trace(self) -> None:
        logZ_trace = []
        cumulative_log_weights = []
        for w in self.log_weights:
            cumulative_log_weights.append(w)
            logZ_estimate = logsumexp(cumulative_log_weights)
            logZ_trace.append(logZ_estimate)
        plt.figure(figsize=(8, 4))
        plt.plot(logZ_trace, lw=2)
        plt.axvline(self.total_iterations, color='r', linestyle='--', label="Final live points")
        plt.xlabel("Iteration")
        plt.ylabel("Estimated log(Z)")
        plt.title("Log-Evidence Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        print(f"Log Evidence: {self.get_log_evidence():.3f}")
        samples, weights = self.get_posterior_samples()
        mean = np.average(samples, axis=0, weights=weights)
        print("Posterior mean:", mean)
