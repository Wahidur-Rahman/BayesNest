# BayesNest

**BayesNest** is a lightweight, log-space nested sampler for Bayesian inference, supporting complex posterior structures including multimodal, curved, and non-Gaussian distributions. It supports multiple constrained sampling strategies including:

- Naive rejection sampling
- Ellipsoidal sampling (MultiNest-style)
- MCMC-constrained sampling
- Slice sampling
- Diffusive sampling (tempered MCMC)

BayesNest computes both posterior samples and Bayesian model evidence

---

## Features

- Log evidence computation
- Posterior sampling with log-weights
- Support for multimodal and curved posteriors
- Corner plot visualization of parameter distributions
- Multiple constrained samplers (`uniform`, `ellipsoid`, `mcmc`, `slice`, `diffusive`)
- Convergence diagnostics via log-evidence trace plots
- Minimal, dependency-light design

---

## Installation

```bash
pip install bayesnest
```

## Examples
All examples are located in the examples/ directory of the repo.
### 1: Linear Gaussian Model


```bash
python examples/linear_gaussian_example.py
```
This example:

- Simulates data from a linear Gaussian model y = ax + b + noise

- Use nested sampling to estimate the posterior distribution over
slope `a` and intercept `b`

- Print the posterior mean and log evidence

- Generate a 2D corner plot of the posterior distribution plot

- Plots the convergence of the log-evidence estimate

### 2: Multimodal Posterior
A synthetic example with two separated Gaussian modes is also found in the github repo

```bash
python examples/multimodal_example.py
```
This synthetic example defines a 2D likelihood with two well-separated Gaussian peaks and:

- Sample using ellipsoidal decomposition

- Reveal both modes clearly in the posterior distribution plot

- Demonstrate ellipsoid nested sampling’s strength in handling multimodal distributions

Expected output:

- A bimodal posterior in the corner plot

- A log-evidence trace that reflects both modes properly being integrated

### 3: Rosenbrock

```bash
python examples/rosenbrock_example.py
```
This classic benchmark stresses the sampler with a narrow, curved valley in 5D.
- Uses slice sampling to effectively explore curved likelihood regions
- Computes posterior samples and log-evidence
- Generates a pairwise corner plot

Expected output:
- The expected 2D plot between theta_0 and theta_1 should show a tight, curved valley, starting from -1 to 1
- Posterior mean near theta_i = 1 for all i



