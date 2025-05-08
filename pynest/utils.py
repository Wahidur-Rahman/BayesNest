import numpy as np
def generate_cube(ndim: int):
    """Generates a point uniformly in the unit cube [0, 1]^ndim."""
    return np.random.rand(ndim)