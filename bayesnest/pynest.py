import inspect
from bayesnest.utils import *
from typing import Callable, Tuple, Optional
import corner
import matplotlib.pyplot as plt
from bayesnest.priors import *
from scipy.special import logsumexp
import time
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

    Attributes
    ----------
    log_evidence : float
        The natural log of the Bayesian evidence (marginal likelihood).
    samples : list
        Collected posterior samples.
    log_weights : list
        Log-weights associated with each sample.
    """
    def __init__(
            self, log_likelihood: Callable,
            prior: Callable,
            ndim: int,
            live_points: int = 5,
            max_iterations: int = 400,
            verbose: bool = False
    ):


        self.log_likelihood: Callable = log_likelihood
        self.prior: Callable = prior
        self.ndim: int = ndim
        self.live_points: int = live_points
        self.max_iterations:int  = max_iterations
        self.verbose = verbose

        #output
        self.log_evidence: float = np.nan
        self.logX: float = 0.0
        self.samples: list = []
        self.log_weights: list = []
        self.log_Z_terms: list = []


        #Verify Inputs
        assert live_points > 2, f"Need at least 2 live_points - you have {live_points}"

        # TODO: Verify the prior function format is correct

        #Generate the unit cube
        unit_cube = generate_cube(ndim)

        #confirm the dimensions of the returned cube
        prior_cube = prior(unit_cube)
        prior_length = len(prior_cube)
        assert prior_length == ndim, f"Your prior function vector length, {prior_length}, is not equal to ndim ({ndim})"
        
        # @TODO: Verify the likelihood function format is correct

    def _logsubexp(self, a, b):
        if b >= a:
            raise ValueError("_logsubexp requires b < a")
        return a + np.log1p(-np.exp(b - a))
    def run(self):
        start = time.time()
        # Initial sample  of points
        unit_cubes = [generate_cube(self.ndim) for i in range(self.live_points)]
        prior_cubes = [self.prior(cube) for cube in unit_cubes]

        # Compute Likelihoods
        self.likelihood_live = np.array([self.log_likelihood(cube) for cube in prior_cubes])
        for i in range(self.max_iterations):
            if self.verbose:
                print("Iteration:",i)
            # Find the lowest likelihood and update evidence samples
            i_min = np.argmin(self.likelihood_live)
            l_min = self.likelihood_live[i_min]
            worst = prior_cubes[i_min]
            logX_new = self.logX - 1.0/self.live_points
            log_dX = self._logsubexp(self.logX, logX_new)
            log_weight = l_min + log_dX
            self.samples.append(worst)
            self.log_weights.append(log_weight)
            self.log_Z_terms.append(log_weight)

            # Replace worst point with a new one
            while True:
                w_new = self.prior(generate_cube(self.ndim))
                if self.log_likelihood(w_new) > l_min:
                    break
            prior_cubes[i_min] = w_new
            self.likelihood_live[i_min] = self.log_likelihood(w_new)
            self.logX = logX_new
        self.total_iterations = self.max_iterations
        # --- Include final live points ---
        log_X_final = -self.total_iterations / self.live_points
        log_dX_final = np.log(np.exp(log_X_final) / self.live_points)

        for i in range(self.live_points):
            log_weight = self.likelihood_live[i] + log_dX_final
            self.samples.append(prior_cubes[i])
            self.log_weights.append(log_weight)
            self.log_Z_terms.append(log_weight)
        self.log_evidence = logsumexp(self.log_Z_terms)
        end = time.time()
        print(f"Sampling Finished in {end-start:.3f} seconds")

    def get_posterior_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        log_weights = np.array(self.log_weights)
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        return np.array(self.samples), weights

    def get_log_evidence(self) -> float:
        return self.log_evidence

    def plot_posterior(self, labels: Optional[list] = None, truths: Optional[list]=None) -> None:
        samples, weights = self.get_posterior_samples()
        if labels is None:
            labels = [f"w{i}" for i in range(self.ndim)]

        corner.corner(
            samples,
            weights=weights,
            labels=labels,
            truths=truths,
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 12}
        )
        plt.show()

    def summary(self):
        print(f"Log Evidence: {self.get_log_evidence():.3f}")
        samples, weights = self.get_posterior_samples()
        mean = np.average(samples, axis=0, weights=weights)
        print("Posterior mean:", mean)
    