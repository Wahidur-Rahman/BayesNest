"""
Example: Nested sampling for a multimodal 2D posterior.
This demonstrates the ability to explore disconnected modes.
"""

import numpy as np
from bayesnest import NestedSampler
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --- Define a bimodal log-likelihood ---
def multimodal_loglike(theta: np.ndarray) -> float:
    mode1 = multivariate_normal(mean=[-2, -2], cov=[[0.5, 0], [0, 0.5]])
    mode2 = multivariate_normal(mean=[2, 2], cov=[[0.5, 0], [0, 0.5]])
    p1 = mode1.pdf(theta)
    p2 = mode2.pdf(theta)
    return np.log(0.5 * p1 + 0.5 * p2 + 1e-300)  # add epsilon to avoid log(0)

# --- Define a uniform prior over a large box ---
def uniform_prior(cube: np.ndarray) -> np.ndarray:
    return -5 + 10 * cube  # maps [0,1]^2 → [-5, 5]^2

# --- Run Nested Sampler ---
sampler = NestedSampler(
    log_likelihood=multimodal_loglike,
    prior=uniform_prior,
    ndim=2,
    live_points=500,
    max_iterations=1000,
    sampler="ellipsoid",
    tolerance=1e-5,
    verbose=True
)

sampler.run()
sampler.summary()

# --- Posterior Plots ---
sampler.plot_posterior(labels=["x", "y"])
sampler.plot_logZ_trace()
