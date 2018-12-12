import numpy as np
import pymc as pm
from matplotlib import pyplot as plt


n = 100 # number of data points
x_data = np.random.normal(0, 1, n)
y_data = x_data + 0.5 + np.random.normal(0, 0.35, n)

std = pm.Uniform("std", 0, 100)  

@pm.deterministic
def prec(U=std):
    return 1.0 / (U) ** 2

beta = pm.Normal("beta", 0, 0.0001)
alpha = pm.Normal("alpha", 0, 0.0001)


@pm.deterministic
def linear_regress(x=x_data, alpha=alpha, beta=beta):
    return x*alpha+beta

y = pm.Normal('y', linear_regress, prec, value=y_data, observed=True)

model = pm.Model([y, std, prec, alpha, beta])
mcmc = pm.MCMC(model)
mcmc.sample(iter=100000, burn=50000, thin=10)


alpha_samples = mcmc.trace('alpha')[:]
beta_samples = mcmc.trace('beta')[:]
std_samples = mcmc.trace('std')[:]

# histogram of the samples:

plt.hist(alpha_samples, normed = True)
plt.show()

plt.hist(beta_samples, normed = True)
plt.show()

plt.hist(std_samples, normed = True)
plt.show()
