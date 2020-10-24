"""Little utility functions used in several classifier sketches.
"""

import sklearn.datasets
import numpy as np

def split_training_test(document_pairs):
    """
    Given a list of things, split them into training and test.
    Returns a pair of lists: training, test.

    Simplest split: every 10th thing is in the test set.
    """
    training = []
    test = []
    for i, pair in enumerate(document_pairs):
        if i % 10 == 9:
            test.append(pair)
        else:
            training.append(pair)
    return training, test

def load_binary_toy_documents():
    out = [
        ("pos", "a".split()),
        ("pos", "a".split()),
        ("pos", "a".split()),
        ("neg", "b".split()),
        ("neg", "b".split()),
        ("neg", "b".split()),
        ("neg", "c".split()),
        ("neg", "c".split()),
        ("neg", "c".split()),
        ("pos", "a".split()),
    ]
    return out

def load_multinomial_toy_documents():
    out = []
    for i in range(30):
        out.append(("A", "b c".split()))
    for i in range(30):
        out.append(("B", "a c".split()))
    for i in range(30):
        out.append(("C", "a b".split()))
    return out

def load_iris_example_pairs():
    out = []
    iris = sklearn.datasets.load_iris()
    for example, label in zip(iris.data, iris.target):
        # tack a 1 on the end here as a bias term.
        out.append((label, np.append(example, [1])))
    return out
