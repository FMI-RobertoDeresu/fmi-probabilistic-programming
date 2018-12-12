import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

tmp = np.loadtxt('crime_train.txt')
x_data = tmp[:, :-1]
y_data = tmp[:, -1]

tmp = np.loadtxt('crime_test.txt')
xt = tmp[:, :-1]
yt = tmp[:, -1]

d = x_data.shape[1]

std = pm.Uniform("std", 0, 100)  

@pm.deterministic
def prec(U=std):
    return 1.0 / (U) ** 2

beta = pm.Normal("beta", 0, 0.0001)


alpha = np.empty(d, dtype=object)
for i in range(d):
    alpha[i] = pm.Normal('alpha_%i' % i, 0, 0.0001)


@pm.deterministic
def linear_regress(x=x_data, alpha=alpha, beta=beta):
    return x.dot(alpha) + beta

y = pm.Normal('y', linear_regress, prec, value=y_data, observed=True)

model = pm.Model([y, std, prec, pm.Container(alpha), beta])
mcmc = pm.MCMC(model)
mcmc.sample(iter=100000, burn=50000, thin=10)

ae = np.empty(d)
for i in range(d):  
    ae[i] = np.mean(mcmc.trace('alpha_%i' % i)[:], axis = 0)  

be = np.mean(mcmc.trace('beta')[:], axis = 0)
print()
print()

yh = xt.dot(ae) + be
print('Yh             Yt    MSE')
for i in range(yt.shape[0]):
    print(yh[i], yt[i], (yh[i] - yt[i]) ** 2)


for i in range(d):  
    alpha_samples = mcmc.trace('alpha_%i' % i)[:]
    plt.hist(alpha_samples)
    plt.title('Posterior of alpha_%i' % i)
    plt.show()


beta_samples = mcmc.trace('beta')[:]
std_samples = mcmc.trace('std')[:]

# histogram of the samples:

#plt.hist(alpha_samples, normed = True)
#plt.show()

plt.hist(beta_samples)
plt.title('Posterior of beta')
plt.show()

plt.hist(std_samples)
plt.show()
