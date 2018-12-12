import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

class Bandits(object):

    """
    This class represents N bandits machines.

    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.

    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """

    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)

    def pull(self, i):
        # i is which arm to pull
        return np.random.rand() < self.p[i]

    def __len__(self):
        return len(self.p)

class BayesianStrategy(object):

    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    """

    def __init__(self, bandits):

        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros(n_bandits)
        self.trials = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.bb_score = []

    def sample_bandits(self, n=1):

        bb_score = np.zeros(n)
        choices = np.zeros(n)

        for k in range(n):
            # sample from the bandits's priors, and select the largest sample

            P0 = pm.Uniform('P0', 0, 1)
            P1 = pm.Uniform('P1', 0, 1)
            P2 = pm.Uniform('P2', 0, 1)

            X0 = pm.Binomial('X0', value = self.wins[0], n = self.trials[0]+1, p = P0, observed = True)
            X1 = pm.Binomial('X1', value = self.wins[1], n = self.trials[1]+1, p = P1, observed = True)
            X2 = pm.Binomial('X2', value = self.wins[2], n = self.trials[2]+1, p = P2, observed = True)

            mcmc0 = pm.MCMC([P0, X0])
            mcmc0.sample(6000, 1000)
            P0_samples = mcmc0.trace('P0')[:]

            mcmc1 = pm.MCMC([P1, X1])
            mcmc1.sample(6000, 1000)
            P1_samples = mcmc1.trace('P1')[:]

            mcmc2 = pm.MCMC([P2, X2])
            mcmc2.sample(6000, 1000)
            P2_samples = mcmc2.trace('P2')[:]

            print();
            print();
            print(k);
            choice = np.argmax([np.random.choice(P0_samples), np.random.choice(P1_samples), np.random.choice(P2_samples)])

            # sample the chosen bandit
            result = self.bandits.pull(choice)

            # update priors and score
            self.wins[choice] += result
            self.trials[choice] += 1
            bb_score[k] = result
            self.N += 1
            choices[k] = choice

        self.bb_score = np.r_[self.bb_score, bb_score]
        self.choices = np.r_[self.choices, choices]
        return

hidden_prob = np.array([0.85, 0.60, 0.75])
bandits = Bandits(hidden_prob)

bayesian_strat = BayesianStrategy(bandits)

bayesian_strat.sample_bandits(3000)

def regret(probabilities, choices):
    w_opt = probabilities.max()
    return (w_opt - probabilities[choices.astype(int)]).cumsum()

_regret = regret(hidden_prob, bayesian_strat.choices)
plt.plot(_regret, lw=3)

plt.title("Total Regret of Bayesian Bandits Strategy (MCMC)")
plt.xlabel("Number of pulls")
plt.ylabel("Regret after $n$ pulls")
plt.show()
