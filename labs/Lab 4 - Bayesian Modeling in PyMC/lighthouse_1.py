import numpy as np
import pymc as pm
from matplotlib import pyplot as plt


# The parameters to be inferred. We only know them here because we are synthesising the data.
true_alpha = 10
true_beta = 50

num_flashes = 5000

# Generate the angles
true_thetas = np.random.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=num_flashes)

# Generate the x coordinates of the flashes along the coastline
data = true_alpha + true_beta * np.tan(true_thetas)


alpha = pm.Normal("alpha", 0, 1.0/50**2)
beta = pm.Exponential("beta", 1.0/100)

# @pm.stochastic(name="obs", dtype=np.float64, observed=True)
# def obs(value=data, alpha=alpha, beta=beta):
#     return np.sum(np.log(beta) - np.log(np.pi) - np.log(beta**2 + (alpha-value)**2))

@pm.observed(name="obs", dtype=np.float64)
def obs(value=data, alpha=alpha, beta=beta):
    return np.sum(np.log(beta) - np.log(np.pi) - np.log(beta**2 + (alpha-value)**2))

model = pm.Model([alpha, beta, obs])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)


alpha_samples = mcmc.trace("alpha")[:]
beta_samples = mcmc.trace("beta")[:]

# histogram of the samples:

plt.hist(alpha_samples, normed = True)
plt.show()

plt.hist(beta_samples, normed = True)
plt.show()
