from pynest import NestedSampler
import numpy as np
mean = 0
dev = 5
def simulate_data(a,b,n):
    x = np.linspace(0,10,n)
    y = a*x + b + np.random.normal(loc=mean,scale=dev,size=n)
    return x,y

x, y = simulate_data(5,10,25)

def log_gaussian(x, mu, sig):
    return np.log(1/np.sqrt(2 * np.pi * sig**2.0))  +  (-0.5/sig**2.0) * (x - mu)**2.0

def myloglike(cube):
    log_likelihood = 0
    for i in tuple(zip(x,y)):
        log_likelihood += log_gaussian(i[1], cube[0]*i[0] + cube[1], dev)
    return log_likelihood

def myprior(cube):
    cube[0] = cube[0] * 10
    cube[1] = cube[1] * 20
    return cube

sampler = NestedSampler(log_likelihood=myloglike, prior=myprior, ndim = 2, max_iterations=50)
sampler.run()
sampler.summary()
sampler.plot_posterior(labels=['Intercept','a','b'], truths=[5,10])
