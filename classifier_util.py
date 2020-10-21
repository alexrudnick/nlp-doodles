"""Little utility functions used in several classifier sketches.
"""

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
