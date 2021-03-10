import inspect
from utils import *
import typing as tp
import numpy as np
from samplers import *
VALID_SAMPLERS = [
    'mcmc',
    'ellipsoidal',
    'dynamic',
    'slice'
]

noise_term = 10e-200
class Sampler:
    def __init__(self, log_likelihood: tp.Callable,
        prior: tp.Callable,
        ndim: int,
        sample_method:str = 'mcmc',
        initial_points: int = 400,
        evidence_tolerance:float = 0.1
        ):


        self.log_likelihood = log_likelihood
        self.prior = prior
        self.ndim = ndim
        self.sample_method = sample_method
        self.intial_points = initial_points
        self.evidence_tolerance = evidence_tolerance
        
        self.evidence = 0
        self.information = 0
        self.posterior = []

        self.X_list = []
        self.output_priors = [] # list of tuples of prior samples
        self.likelihood_output_chain = [] #A corresponding set of likelihoods to match those in the output chain
        self.likelihood_live = [] #Likelihoods for current set of samples

        assert sample_method in VALID_SAMPLERS, f"Invalid sample_method: {sample_method}"

        assert initial_points > 2, f"Need at least 2 initial_points - you have{initial_points}"

        # Verify the prior function format is correct
        
        #Generate the unit cube
        unit_cube = generate_cube(ndim)

        #confirm the dimensions of the returned cube
        prior_cube = prior(unit_cube)
        prior_length = len(prior_cube)
        assert prior_length == ndim, "Your prior function vector length, {prior_length}, is not equal to ndim ({ndim})"
        
        # @TODO: Verify the likelihood function format is correct
         
    
        

    def sample(self):
        #Steps to sampling
        #Step 1 - Initial sample  of points
        unit_cubes = [generate_cube(self.ndim) for i in range(self.intial_points)]
        prior_cubes = [self.prior(cube) for cube in unit_cubes]


        X_count = 1
        X = np.exp(-X_count/self.intial_points)
        X_prev = 1

        self.X_list.append(X_prev)
        self.likelihood_output_chain.append(0)



        #Step 2 - Compute Likelihoods
        self.likelihood_live = np.array([self.log_likelihood(cube) for cube in prior_cubes])
        l_max = max(self.likelihood_live)
        #Evidence acurracy tolerance
        while l_max*(X - X_prev) > self.evidence_tolerance:
            #Step 3 - Discard lowest likelihood
            min_loc = np.where(self.likelihood_live == min(self.likelihood_live))[0][0]
            likelihood_lower_bound = self.likelihood_live[min_loc]

            unit_prior_lower_bound = unit_cubes[min_loc]
            prior_lower_bound = prior_cubes[min_loc]
            
            #Append to samples list
            self.likelihood_output_chain.append(likelihood_lower_bound)
            self.X_list.append(X)

            #Step 4 - replace discarded point from samples from prior
            new_sample = mcmc_proposal(prior_lower_bound,likelihood_lower_bound)
            new_likelihood_val = self.log_likelihood(new_sample)
            while new_likelihood_val < likelihood_lower_bound:
                new_sample = mcmc_proposal(prior_lower_bound,likelihood_lower_bound)
                new_likelihood_val = self.log_likelihood(new_sample)
            prior_cubes[min_loc] = new_sample #overwrite
            X_count += 1
            X_prev = X
            X = np.exp(-X_count/self.intial_points)
            self.likelihood_live[min_loc] = self.log_likelihood(new_sample)
            
            break

        #Step 6 - if terminating, compute posterior, evidence and information
        return

    def return_chains():
        return self.X_list, self.likelihood_output_chain
    