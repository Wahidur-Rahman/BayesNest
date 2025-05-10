"""
Example: High-dimensional Rosenbrock likelihood to stress test sampling on curved valleys.
"""

import numpy as np
from bayesnest import NestedSampler

# --- Log-likelihood: n-dimensional Rosenbrock valley ---
def rosenbrock_loglike(theta: np.ndarray, sigma: float = 0.1) -> float:
    theta = np.asarray(theta)
    val = np.sum(100.0 * (theta[1:] - theta[:-1]**2)**2 + (1 - theta[:-1])**2)
    return -val / (2 * sigma**2)

# --- Uniform prior over [-5, 5]^d ---
def uniform_prior(cube: np.ndarray) -> np.ndarray:
    return -5 + 10 * cube

# --- Settings ---
ndim = 5
live_points = 500

# --- Run the Nested Sampler ---
sampler = NestedSampler(
    log_likelihood=rosenbrock_loglike,
    prior=uniform_prior,
    ndim=ndim,
    sampler="slice",
    live_points=live_points,
    max_iterations=10000,
    slice_steps=15,
    step_scale=0.3,
    tolerance=1e-4,
    verbose=True
)

sampler.run()
sampler.summary()
labels = [f"$\\theta_{i}$" for i in range(ndim)]
truths = [1.0] * ndim

sampler.plot_posterior(labels=labels, truths=truths)

sampler.plot_logZ_trace()

