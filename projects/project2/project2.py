from .binary_classification import run as run_binary_classification
from .multiclass_classification import run as run_multiclass_classification


def run():
    run_binary_classification()
    run_multiclass_classification()
