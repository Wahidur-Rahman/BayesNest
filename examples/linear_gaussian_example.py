"""
Example: Bayesian linear regression with analytical log-evidence comparison.
"""

import numpy as np
import scipy.stats
from bayesnest import NestedSampler

# --- Simulated Data ---
def simulate_data(a: float, b: float, n: int, noise_std: float):
    x = np.linspace(0, 10, n)
    y = a * x + b + np.random.normal(scale=noise_std, size=n)
    return x, y

true_a, true_b, n = 5, 10, 25
sigma = 1.0  # Observation noise std dev
x, y = simulate_data(true_a, true_b, n, sigma)

# --- Analytical Posterior and Log Evidence ---
X = np.vstack([x, np.ones_like(x)]).T
y_col = y.reshape(-1, 1)

prior_mean = np.array([5.0, 10.0])
prior_std = np.array([0.5, 0.5])
Sigma_0 = np.diag(prior_std**2)
Sigma_0_inv = np.linalg.inv(Sigma_0)
sigma2 = sigma**2

Sigma_post = np.linalg.inv(Sigma_0_inv + (X.T @ X) / sigma2)
mu_post = Sigma_post @ (Sigma_0_inv @ prior_mean.reshape(-1, 1) + (X.T @ y_col) / sigma2)

Sigma_y = sigma2 * np.eye(n) + X @ Sigma_0 @ X.T
resid = y_col - X @ prior_mean.reshape(-1, 1)
_, logdet = np.linalg.slogdet(Sigma_y)

logZ_analytic = (
    -0.5 * n * np.log(2 * np.pi)
    -0.5 * logdet
    -0.5 * resid.T @ np.linalg.inv(Sigma_y) @ resid
).item()

print("Analytical posterior mean (mu_post):", mu_post.ravel())
print("Analytical log evidence (logZ):", logZ_analytic)

# --- Define Prior and Log-Likelihood ---
def log_gaussian(x, mu, sig):
    return -0.5 * np.log(2 * np.pi * sig**2) - 0.5 * ((x - mu) / sig)**2

def myprior(cube: np.ndarray) -> np.ndarray:
    cube = np.clip(cube, 1e-10, 1 - 1e-10)  # prevent infs
    return prior_mean + prior_std * scipy.stats.norm.ppf(cube)

def myloglike(theta: np.ndarray) -> float:
    log_likelihood = 0.0
    for xi, yi in zip(x, y):
        mu = theta[0] * xi + theta[1]
        log_likelihood += log_gaussian(yi, mu, sigma)

    log_prior = -0.5 * np.sum(((theta - prior_mean) / prior_std) ** 2)
    log_prior -= np.sum(np.log(prior_std)) + len(theta) * 0.5 * np.log(2 * np.pi)
    return float(log_likelihood + log_prior)

# --- Run Nested Sampler ---
sampler = NestedSampler(
    log_likelihood=myloglike,
    prior=myprior,
    ndim=2,
    live_points=200,
    max_iterations=100,
    sampler="ellipsoid",
    tolerance=1e-5,
    verbose=True
)
sampler.run()
sampler.summary()

# --- Plots ---
sampler.plot_posterior(labels=['Slope (a)', 'Intercept (b)'], truths=[true_a, true_b])
sampler.plot_logZ_trace()
