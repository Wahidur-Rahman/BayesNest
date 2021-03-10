import sys
from pynest import Sampler
import numpy as np

def myloglike(cube):
    cube = np.array(cube)
    return (np.cos(cube / 2.)).prod()

def myprior(cube):
    cube[0] = cube[0] * 10
    cube[1] = cube[1] * 10
    return cube

def test_mcmc_instantation():
    sampler = Sampler(log_likelihood=myloglike, prior=myprior, ndim = 2, sample_method = 'mcmc')
    sampler.sample()
