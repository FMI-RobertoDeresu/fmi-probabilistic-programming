from .input import input_data
from .lda import LDA
from multiprocessing import Process, Lock


def run():
    lock = Lock()
    for i in range(len(input_data["fit"])):
        Process(target=runt_fit_sample, args=(lock, input_data["fit"][i])).start()


def runt_fit_sample(lock, fit_settings):
    lda = LDA(**input_data["tests"][input_data["test_to_run"]])
    lda.fit(**fit_settings)

    lock.acquire()
    print(str(fit_settings) + "\n")
    print_results(lda)
    print(65 * "=" + "\n")
    lock.release()


def print_results(lda):
    print("Docs as observable")
    for doc in lda.docs_as_observable:
        print(doc)
    print()

    print("Topics distribution per document")
    topics_distribution_per_doc = lda.get_topics_distribution_per_doc()
    for doc_topics_distribution in topics_distribution_per_doc:
        print(doc_topics_distribution)
    print()

    print("Words distribution per topic")
    words_distribution_per_topic = lda.get_words_distribution_per_topic()
    for topic_words_distribution in words_distribution_per_topic:
        print(topic_words_distribution)
    print()

    print("Topics assignments")
    topics_assignments = lda.get_topics_assignments()
    for doc_topics_assignment in topics_assignments:
        print(doc_topics_assignment)
    print()

    print("Docs similarity")
    docs_similarity = lda.get_docs_similarity()
    for doc_similarity in docs_similarity:
        print(doc_similarity)
    print()

