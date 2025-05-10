# BayesNest

**BayesNest** is a lightweight, log-space nested sampler for Bayesian inference, supporting multimodal and non-Gaussian posteriors. It includes both naive rejection sampling and ellipsoidal constrained sampling (inspired by MultiNest), and computes both posterior samples and Bayesian model evidence.

---

## Features

- Log-space evidence computation (numerically stable)
- Posterior sampling with log-weights
- Support for multimodal inference via ellipsoidal sampling
- Corner plot visualization of posteriors
- Convergence diagnostics for log-evidence
- Simple, extensible Python codebase

---

## Installation

From PyPI (after publishing):

```bash
pip install bayesnest
```
## Example: Linear Gaussian Model
Once installed, you can run the examples found in the github repo

```bash
python examples/linear_gaussian_example.py
```
This will:

- Simulate data from a linear Gaussian model

- Use nested sampling to estimate the posterior over
slope `a` and intercept `b`

- Print the posterior mean and log evidence

- Generate a 2D corner plot of the posterior

- Show the evolution of the log-evidence estimate over iterations

## Example: Multimodal Posterior
A synthetic example with two separated Gaussian modes is also found in the github repo

```bash
python examples/multimodal_example.py
```
This example will:

- Define a 2D likelihood with two disjoint Gaussian peaks

- Sample using ellipsoidal decomposition

- Reveal both modes in the posterior

- Demonstrate ellpisoid nested sampling’s strength in handling multimodal distributions

Expected output:

- A bimodal posterior in the corner plot

- A log-evidence trace that reflects both modes being integrated


