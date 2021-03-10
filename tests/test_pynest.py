import sys
sys.path.append('../')
sys.path.append('./')
from pynest import Sampler

def myloglike():
    return

def myprior():
    return

sampler = Sampler(log_likelihood=myloglike, prior=myprior,sample_method = 'test')