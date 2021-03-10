import numpy as np
import pandas as pd
import typing as tp
import random

def mcmc_proposal(starting_point:tp.List[float],step_size:float):
    rand_int = random.randint(0,len(starting_point)-1)
    sample = starting_point[rand_int] + random.uniform(0,1)*step_size
    #print('proposing a new point')
    while sample > 1 or sample <0:
        sample = starting_point[rand_int] + random.uniform(0,1)*step_size
        #print(starting_point[rand_int],sample)
    starting_point[rand_int] = sample
    #print('proposal',sample)
    return starting_point