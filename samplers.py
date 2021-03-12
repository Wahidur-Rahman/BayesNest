import numpy as np
import pandas as pd
import typing as tp
import random

mcmc_scale = 0.1
def mcmc_proposal(starting_point:tp.List[float]):
    rand_int = random.randint(0,len(starting_point)-1)
    sample = np.random.normal(loc = starting_point[rand_int] , scale = mcmc_scale)
    #sample = starting_point[rand_int] + step

    while sample > 1 or sample <0:
        sample = np.random.normal(loc = starting_point[rand_int] , scale = mcmc_scale)
        #sample = starting_point[rand_int] + step
        print('resampling',sample)

    starting_point[rand_int] = sample
    return starting_point