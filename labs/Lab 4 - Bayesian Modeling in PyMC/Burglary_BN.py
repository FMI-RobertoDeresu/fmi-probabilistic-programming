import pymc as pm
import numpy as np

B = pm.Bernoulli("B", 0.001)
E = pm.Bernoulli("E", 0.002)

p_A = pm.Lambda("p_A", lambda B = B, E = E: np.where(B, np.where(E, .95, .94), np.where(E, .29, 0.001)))
A = pm.Bernoulli("A", p_A)

p_J = pm.Lambda("p_J", lambda A = A: np.where(A, .9, .05))
J = pm.Bernoulli('J', p_J, value = [1], observed = True)

p_M = pm.Lambda("p_M", lambda A = A: np.where(A, .7, .01))
M = pm.Bernoulli('M', p_M, value = [1], observed = True)

model = pm.Model([B, E, A, J, M])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)
B_samples = mcmc.trace('B')[:]

print()
print()
print("Burglary probability:", B_samples.mean())



