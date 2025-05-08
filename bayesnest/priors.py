import numpy as np
from scipy.special import erfinv,gamma

# log-uniform prior
# given point sampled from unif(0, 1)
# return point sampled on a uniform log scale from min_val to max_val
def log_uniform(r, min_val, max_val):
    point = 0
    if (r <= 0):
        point = -1.0
    else:
        log_min_val = np.log10(min_val)
        log_max_val = np.log10(max_val)
        point = 10.0 ** (log_min_val + r * (log_max_val - log_min_val))
    return point

def log_invgamma(x, a, b):
    if x < 1e-3:
        return -1e90
    return np.log(b**a / gamma(a) * x**(-a - 1) * np.exp(-b /x))


# gaussaian prior
# given point sampled from unif(0, 1)
# return point sampled from gaussian with (mu, sigma)
def gaussian(r, mu, sigma):
    point = 0
    if (r < 1e-16 or 1 - r < 1e-16):
        point = -1.0
    else:
        point = mu + sigma * np.sqrt(2) * erfinv(2 * r - 1)
    return point
