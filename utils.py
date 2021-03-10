import random

def generate_cube(ndim):
    # Generates a cube of length ndim
    return [random.uniform(0,1) for i in range(ndim)]