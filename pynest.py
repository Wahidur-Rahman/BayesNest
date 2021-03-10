import inspect
import typing as tp

VALID_SAMPLERS = [
    'mcmc',
    'ellipsoidal',
    'dynamic_nested',
    'slice'
]

class Sampler:
    def __init__(self, log_likelihood: tp.Callable ,prior: tp.Callable,sample_method:str = 'mcmc'):
        self.log_likelihood = log_likelihood
        self.prior = prior
        self.sample_method = sample_method
        assert sample_method in VALID_SAMPLERS, f"Invalid sample_method: {sample_method}"
        print(f'Using {sample_method} sampler')
    
    def sampler():
        return
    