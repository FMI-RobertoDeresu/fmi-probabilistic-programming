import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

N = 2500

#create some artificial data. 
lifetime = pm.rweibull( 2, 5, size = N ) 
birth = pm.runiform(0, 10, N)

censor = (birth + lifetime) > 10 #an individual is right-censored if this is True 
lifetime_ = lifetime.copy()
lifetime_[censor] = 10 - birth[censor] #we only see this part of their lives.

#this begins the model 
alpha = pm.Uniform("alpha", 0,20) 
#lets just use uninformative priors 
beta = pm.Uniform("beta", 0,20) 

@pm.observed
def survival(value=lifetime_, alpha = alpha, beta = beta ):
    return np.sum( (1-censor) * (np.log( alpha/beta) + (alpha-1)*np.log(value/beta) - (value/beta)**(alpha)))

mcmc = pm.MCMC([alpha, beta, survival ] )
mcmc.sample(50000, 30000)


alpha_samples = mcmc.trace('alpha')[:]
beta_samples = mcmc.trace('beta')[:]

# histogram of the samples:

plt.hist(alpha_samples, normed = True)
plt.show()
plt.hist(beta_samples, normed = True)
plt.show()

medianlifetime_samples = beta_samples * (np.log(2)**(1 / alpha_samples))

plt.hist(medianlifetime_samples, normed = True)
plt.show()
