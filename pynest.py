import inspect
from utils import *
import typing as tp
import numpy as np
from samplers import *
from random import randint
import copy
VALID_SAMPLERS = [
    'mcmc',
    'ellipsoidal',
    'dynamic',
    'slice'
]
# Useful function
def logsumexp(values):
  biggest= np.max(values)
  x = values - biggest
  #print('arrs',values,biggest,x)
  result = np.log(np.sum(np.exp(x))) + biggest
  #result = sum(np.exp(values))
  #print('result',result)
  return result

noise_term = 10e-200
class Sampler:
    def __init__(self, log_likelihood: tp.Callable,
        prior: tp.Callable,
        ndim: int,
        sample_method:str = 'mcmc',
        initial_points: int = 5,
        evidence_tolerance:float = 0.1
        ):


        self.log_likelihood = log_likelihood
        self.prior = prior
        self.ndim = ndim
        self.sample_method = sample_method
        self.initial_points = initial_points
        self.evidence_tolerance = evidence_tolerance
        
        self.log_evidence = 0
        self.information = 0
        self.posterior = []
        self.step_size = 0.001
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
        assert prior_length == ndim, f"Your prior function vector length, {prior_length}, is not equal to ndim ({ndim})"
        
        # @TODO: Verify the likelihood function format is correct
         

    def sample(self):
        #Steps to sampling
        #Step 1 - Initial sample  of points
        unit_cubes = [generate_cube(self.ndim) for i in range(self.initial_points)]
        prior_cubes = [self.prior(cube) for cube in copy.deepcopy(unit_cubes)]
        print('cube', unit_cubes[0],prior_cubes[0])

        X_count = 1
        logX = -X_count/self.initial_points
        logX_prev = 0

        self.X_list.append(logX_prev)
        self.likelihood_output_chain.append(0)


        #Step 2 - Compute Likelihoods
        self.likelihood_live = np.array([self.log_likelihood(cube) for cube in copy.deepcopy(prior_cubes)])
        #print(self.likelihood_live)
        l_max = max(self.likelihood_live)
        max_evidence_change = l_max + logX_prev
        print('Initial Upper Bound',max_evidence_change)
        #Evidence acurracy tolerance
        counter = 0
        while np.abs(max_evidence_change) > self.evidence_tolerance:
            print(np.abs(max_evidence_change),counter)
            #Step 3 - Discard lowest likelihood
            min_loc = np.where(self.likelihood_live == min(self.likelihood_live))[0][0]
            likelihood_lower_bound = self.likelihood_live[min_loc]
            print('min and max',likelihood_lower_bound,l_max)

            weight = np.log(np.exp(logX_prev) - np.exp(logX))
            increment = likelihood_lower_bound + weight
            self.log_evidence  +=  increment

            unit_prior_lower_bound = unit_cubes[min_loc]
            prior_lower_bound = prior_cubes[min_loc]
            
            #Append to samples list
            self.likelihood_output_chain.append(likelihood_lower_bound)
            self.X_list.append(logX)

            #Step 4 - replace discarded point from samples from prior
            #self.step_size = self.step_size * 0.99 # geometric shrinking of step size
            new_unit_sample = generate_cube(self.ndim)#mcmc_proposal(unit_prior_lower_bound)
            new_prior_sample = self.prior(copy.deepcopy(new_unit_sample))
            #print('new',new_unit_sample,new_prior_sample)
            new_likelihood_val = self.log_likelihood(new_prior_sample)

            while new_likelihood_val < likelihood_lower_bound:
                #print('sampling...')
                new_unit_sample = generate_cube(self.ndim)#mcmc_proposal(unit_prior_lower_bound)
                new_prior_sample = self.prior(copy.deepcopy(new_unit_sample))
                #print('new',new_unit_sample,new_prior_sample)
                new_likelihood_val = self.log_likelihood(new_prior_sample)

            #overwrite the old discarded sample
            unit_cubes[min_loc] = new_unit_sample
            prior_cubes[min_loc] = new_prior_sample 
            self.likelihood_live[min_loc] = new_likelihood_val
            
            #Assigning a new X but keeping track of the previous
            X_count += 1
            logX_prev = logX
            logX = -X_count/self.initial_points
            l_max = max(self.likelihood_live)
            max_evidence_change = l_max + logX_prev
            print('New upper bound',max_evidence_change,self.log_evidence)

            self.likelihood_live[min_loc] = self.log_likelihood(new_prior_sample)
            counter += 1
            if counter ==150:
                break
        
        #Step 6 - if terminating, compute posterior, evidence and information
        #Add up remaining live points
        #increment = logX - np.log(self.intial_points) + logsumexp(self.likelihood_live)
        sorted_l = sorted(self.likelihood_live)
        for i in range(len(self.likelihood_live)):
            l = sorted_l[i]
            weight = np.exp(logX_prev) - np.exp(logX)
            increment = l + np.log(weight)
            self.log_evidence += increment
            X_count += 1
            logX_prev = logX
            logX = -X_count/self.initial_points


        print(increment)
        print(self.log_evidence)
        #self.log_evidence = logsumexp([self.log_evidence, increment])

        print('Log Evidence:',self.log_evidence)
        
        return

    def return_chains():
        return self.X_list, self.likelihood_output_chain
    