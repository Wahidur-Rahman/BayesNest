import sys
from pynest import Sampler
import numpy as np
def myloglike(prior_cube):
    return np.random.randn()

def myprior(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2
    cube[2] = cube[2]*3
    cube[3] = cube[3]*4
    cube[4] = cube[4]*5
    return cube

def test_mcmc_instantation():
    sampler = Sampler(log_likelihood=myloglike, prior=myprior, ndim = 5, sample_method = 'mcmc')
    sampler.sample()
