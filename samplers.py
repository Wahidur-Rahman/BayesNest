import numpy as np
import pandas as pd
import typing as tp
def mcmc_proposal(starting_point:tp.List[float],likelihood_threshold:float):
    sample = starting_point
    return sample