#!/usr/bin/env python3

"""
Demo for how to use the NLTK Naive Bayes classifiers.
It's pretty straightforward to use other NLTK classifiers with the same
features.

Here we use the NaiveBayesClassifier, but for example, you could do...
classifier = nltk.classify.MaxentClassifier.train(training_instances)

... but that's very slow. For a faster implementation, consider using
nltk.classify.SklearnClassifier with sklearn.linear_model.LogisticRegression.
"""

import nltk
from collections import defaultdict
from nltk.corpus import movie_reviews

def data_init():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/movie_reviews")
    except LookupError:
        nltk.download("movie_reviews")
        nltk.download("punkt")

def load_movie_documents():
    """
    Returns a list where each entry is a pair (cat, [token, ...])
    """
    pairs = []
    for fileid in nltk.corpus.movie_reviews.fileids():
        # get the raw text
        text = movie_reviews.raw(fileid)
        category = "pos" if fileid.startswith("pos/") else "neg"

        document = []
        # We can just split the string on spaces, since it's already been
        # preprocessed.
        document = text.split()
        pairs.append((category, document))
    return pairs

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
            training.append(pair)
        else:
            test.append(pair)
    return training, test

def extract_features(tokens):
    """Returns a dictionary from features to counts.

    The dictionary ends up containing word counts, looking like:
    {'cat': 27, 'elephant': 12, 'the': 1002, }
    """
    out = defaultdict(int)
    for tok in tokens:
        out[tok] += 1
    return out

def main():
    data_init()
    document_pairs = load_movie_documents()

    training, test = split_training_test(document_pairs)
    training_instances = [(extract_features(tokens), c)
                          for (c, tokens) in training]
    classifier = nltk.classify.NaiveBayesClassifier.train(training_instances)
    print("TRAINED!!")

    ncorrect = 0
    for (true_c, tokens) in test:
        features = extract_features(tokens)
        guess = classifier.classify(features)
        if guess == true_c:
            ncorrect += 1
    print()
    print("FINAL ACCURACY: {} / {} = {:0.2f}".format(
          ncorrect, len(test), ncorrect / len(test)))

if __name__ == "__main__": main()
