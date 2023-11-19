import numpy as np
from scipy.stats import uniform
from fenics import *
from main2 import forward

# Latin hypercube sampling(Tecnica de muestreo aleatoria)
def latin_hypercube_samples(n_samples, parameter_ranges):
    samples = np.zeros((n_samples, len(parameter_ranges)))
    for i in range(len(parameter_ranges)):
        samples[:, i] = uniform(loc=parameter_ranges[i][0], scale=parameter_ranges[i][1]).rvs(n_samples)
    return samples

parameter_ranges = [(5.0, 2000.0), (5.0, 1000.0)] 
parameter_samples = latin_hypercube_samples(n_samples=100, parameter_ranges=parameter_ranges)

dataset = []
for params in parameter_samples:
    [u, d, e] = forward(params[0], params[1])  # Your forward function
    dataset.append({'params': params, 'temperature_distribution': u.vector().get_local()})

np.savez('heat_transfer_data.npz', dataset=dataset)
