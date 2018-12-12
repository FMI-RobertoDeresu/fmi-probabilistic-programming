import numpy as np
import pymc as pm
from matplotlib import pyplot as plt


true_N = 500
D = pm.rdiscrete_uniform(1, true_N, size = 10)

N = pm.DiscreteUniform("N", lower=D.max(), upper=10000)

observation = pm.DiscreteUniform("obs", lower=0, upper=N, value=D, observed=True)

model = pm.Model([observation, N])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

N_samples = mcmc.trace('N')[:]

# histogram of the samples:

plt.hist(N_samples, normed = True)
plt.show()
