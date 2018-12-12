import numpy as np
import pymc as pm
import networkx as nx
from matplotlib import pyplot as plt

alpha = 0.5
beta = 0.1
L= 9.0

G0 = nx.Graph()

for i in range(1, 10):
    for j in range(i + 1, 11):
        G0.add_edge(i, j)

#G0.add_path(range(1, 11))

#G0.add_path(range(1, 11))
#G0.remove_edge(2, 3)
#G0.remove_edge(3, 4)
#G0.add_edge(2, 4)
#G0.add_edge(3, 7)
#G0.add_edge(8, 10)

# nx.draw(G0, with_labels=True, font_weight='bold')
# plt.show()


@pm.stochastic(dtype=nx.Graph)
def cwg(value = G0, alpha = alpha, beta = beta, L = L):
    tmp = 0
    for i in range(1, len(value)):
        for j in range(i + 1, len(value)+1):
            if value.has_edge(i, j):
                tmp += np.log(alpha) - ((j - i) / (beta * L))
            else:
                tmp += np.log(1 - alpha * np.exp((i - j) / (beta * L)))
    return tmp


class CWGMetropolis(pm.Metropolis):
    """ A PyMC Step Method that walks on connected Waxman Graphs by
        choosing two distinct nodes at random and considering the 
        possible link between them. If the link is already in the
        graph, it consider it for deletion, and if the link is not in
        the graph, it consider it for inclusion, keeping it with the
        appropriate Metropolis probability (no Hastings factor necessary,
        because the chain is reversible, right?)

    """
    def __init__(self, stochastic):
        # Initialize superclass
        pm.Metropolis.__init__(self, stochastic, scale=1., verbose=0, tally=False)

    def propose(self):
        """ Add an edge or remove an edge"""
        G = self.stochastic.value

        G.u_new = np.random.choice(G.nodes()); G.v_new = np.random.choice(G.nodes())
        while G.u_new == G.v_new:
            G.v_new = np.random.choice(G.nodes())

        if G.has_edge(G.u_new, G.v_new):
            G.remove_edge(G.u_new, G.v_new)
            if not nx.is_connected(G):
                G.add_edge(G.u_new, G.v_new)
        else:
            G.add_edge(G.u_new, G.v_new)
        self.stochastic.value = G

    def reject(self):
        """ Restore the graph"""
        G = self.stochastic.value
        if G.has_edge(G.u_new, G.v_new):
            G.remove_edge(G.u_new, G.v_new)
        else:
            G.add_edge(G.u_new, G.v_new)
        self.rejected += 1
        self.stochastic.value = G

@pm.deterministic
def average_degree(G = cwg):
#    return np.sum([t[1] for t in list(G.degree())]) / len(G)
    return np.sum(list(G.degree().values())) / len(G)

mcmc = pm.MCMC([cwg, average_degree])
mcmc.use_step_method(CWGMetropolis, cwg)
mcmc.sample(100000)

avgd_samples = mcmc.trace("average_degree")[:]

plt.hist(avgd_samples[90000:])
plt.show()

nx.draw(cwg.value, with_labels=True, font_weight='bold')
plt.show()

mcmc.sample(100)

nx.draw(cwg.value, with_labels=True, font_weight='bold')
plt.show()

mcmc.sample(100)

nx.draw(cwg.value, with_labels=True, font_weight='bold')
plt.show()
