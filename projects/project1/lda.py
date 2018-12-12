import numpy as np
import pymc as pm
from scipy import spatial


def _word_dict_form_docs(docs):
    word_dict = {}
    word_id = 0
    for doc in docs:
        words = doc.split()
        for word in words:
            if word not in word_dict:
                word_dict[word] = word_id
                word_id += 1

    return word_dict


def _observable_from_word_dict(word_dict, docs):
    observable_docs = []
    for doc in docs:
        observable_doc = []
        words = doc.split()
        for word in words:
            observable_doc.append(word_dict.get(word, 0))
        observable_docs.append(observable_doc)

    return np.array(observable_docs)


class LDA:
    def __init__(self, docs, num_of_topics):
        self.random = np.random.RandomState(7)

        self.word_dict = _word_dict_form_docs(docs)
        self.docs_as_observable = _observable_from_word_dict(self.word_dict, docs)

        self.num_of_topics = num_of_topics
        self.nun_of_words = len(self.word_dict)
        self.num_of_docs = len(self.docs_as_observable)

        alpha = np.ones(self.num_of_topics)
        beta = np.ones(self.nun_of_words)

        len_of_docs = [len(doc) for doc in self.docs_as_observable]

        # topic distribution per-document
        theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i,
                                                    pm.Dirichlet("theta2_%s" % i, theta=alpha))
                              for i in range(self.num_of_docs)])

        # word distribution per-topic
        phi = pm.Container([pm.CompletedDirichlet("phi_%s" % j,
                                                  pm.Dirichlet("phi2_%s" % j, theta=beta))
                            for j in range(self.num_of_topics)])

        #  topic assignments
        z = pm.Container([pm.Categorical("z_%i" % d,
                                         p=theta[d],
                                         size=len_of_docs[d],
                                         value=self.random.randint(self.num_of_topics, size=len_of_docs[d]))
                          for d in range(self.num_of_docs)])

        # word generated from phi, given a topic z
        w = pm.Container([pm.Categorical("w_%i_%i" % (d, i),
                                         p=pm.Lambda("phi_z_%i_%i" % (d, i),
                                                     lambda z=z[d][i], phi=phi: phi[z]),
                                         value=self.docs_as_observable[d][i],
                                         observed=True)
                          for d in range(self.num_of_docs) for i in range(len_of_docs[d])])

        self.model = pm.Model([theta, phi, z, w])
        self.mcmc = pm.MCMC(self.model)

    def fit(self, iterations, burn, thin):
        self.mcmc.sample(iter=iterations, burn=burn, thin=thin)

    def get_topics_distribution_per_doc(self):
        topics_distribution_per_doc = \
            np.array([self.mcmc.trace("theta_%d" % doc_index)[-1] for doc_index in range(self.num_of_docs)])
        return topics_distribution_per_doc

    def get_words_distribution_per_topic(self):
        words_distribution_per_topic = \
            np.array([self.mcmc.trace("phi_%d" % topic_index)[-1] for topic_index in range(self.num_of_topics)])
        return words_distribution_per_topic

    def get_topics_assignments(self):
        topics_assignments = \
            np.array([self.mcmc.trace("z_%d" % doc_index)[-1] for doc_index in range(self.num_of_docs)])
        return topics_assignments

    def get_docs_similarity(self):
        topics_distribution = self.get_topics_distribution_per_doc()
        similarity = np.array([[
            (1 - spatial.distance.cosine(topics_distribution[doc1], topics_distribution[doc2]) if doc1 != doc2 else 0)
            for doc2 in range(self.num_of_docs)] for doc1 in range(self.num_of_docs)])
        return similarity
