import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

N = 20

#create some artificial data. 
lifetime = pm.rweibull( 2, 5, size = N ) 
birth = pm.runiform(0, 10, N)

censor = (birth + lifetime) > 10 #an individual is right-censored if this is True 
lifetime_ = np.ma.masked_array( lifetime, censor ) #create the censorship event. 
lifetime_.set_fill_value( 10 ) #good for computations later.

#this begins the model 
alpha = pm.Uniform("alpha", 0,20) 
#lets just use uninformative priors 
beta = pm.Uniform("beta", 0,20) 
obs = pm.Weibull( 'obs', alpha, beta, value = lifetime_, observed = True )

@pm.potential
def censor_factor(obs=obs): 
    if np.any((obs + birth < 10)[lifetime_.mask] ): 
        return -np.inf
    else:
        return 0

#perform Markov Chain Monte Carlo - see chapter 3 of BMH
mcmc = pm.MCMC([alpha, beta, obs, censor_factor ] )
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
