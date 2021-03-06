#!/usr/bin/env python3

"""
This should do the same thing as simple_bayes_classifier, except that we're
doing logistic regression rather than a Naive Bayes classifier.

Here examples are just numpy arrays, where the final entry in the array is
always 1.0, so we can have a bias term.

Some things to try out:
  - be smarter about SGD: terminate when we're making small moves
  - mini-batches
  - just get the gradient for the whole test set at once
  - different features
    - try using a sentiment lexicon
"""

import copy
import random
import nltk
import numpy as np

from collections import defaultdict
from collections import Counter
from math import log2
from nltk.corpus import movie_reviews

import classifier_util

# adding regularization, are we?
LAMBDA_2 = 1e-5
LAMBDA_1 = 1e-5

LEARNING_RATE = 0.005

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

def extract_vocabulary(D):
    """
    D is the list of documents, pairs of the form (cat, [token, ...])

    Returns vocabulary V as a map from words to word indices.
    """
    vocab = set()
    for (cat, tokens) in D:
        # naively including every word in the vocabulary!
        # This could be trimmed down, or we could do some preprocessing, maybe
        # ignore rare words.
        vocab.update(tokens)

    # oooooh, look at this guy, writing a dictionary comprehension.
    return {wordtype : i for (i, wordtype) in enumerate(sorted(list(vocab)))}

def tokens_to_example(tokens, V):
    """
    Returns a vector of size V+1, bag-of-words model style where the final entry
    is the bias term, which will always be 1.
    """
    example = np.zeros(len(V) + 1)
    example[len(V)] = 1
    for tok in tokens:
        if tok in V:
            index = V[tok]
            example[index] += 1
    return example

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y, yhat):
    """
    Given the true y value (0 or 1) and the estimated one (output of the maxent
    classifier), compute the loss.

    Based on Equation 5.10 in SLP3.
    """
    return -(y * log2(yhat) + (1 - y) * log2(1 - yhat))

def gradient_for_one_example(example, weights, y):
    """
    Given an example vector and a weights vector and the true class (1 or 0) for
    that example, return the gradient vector for that example.

    This is the partial derivative of the loss function with respect to each
    particular weight parameter. So the output vector will be the same size as
    the weight vector, and also the same size as the example vector.
    """

    difference = sigmoid(example.dot(weights)) - y
    l2_penalty = (2 * LAMBDA_2 * weights)
    l1_penalty = (LAMBDA_1 * np.sign(weights))

    gradient = example * difference + l1_penalty + l2_penalty
    return gradient

def new_weights_for_one_example(example, weights, y):
    """
    Returns the next time step's weights, updated for one example.
    """
    gradient = gradient_for_one_example(example, weights, y)
    new_weights = weights - LEARNING_RATE * gradient
    return new_weights

def simple_SGD(training_pairs, initial_weights, max_iters=(100 * 1000)):
    """Given a list of training examples (label, vector) return some weights."""
    weights = np.copy(initial_weights)

    my_training_pairs = copy.copy(training_pairs)
    random.shuffle(my_training_pairs)

    for iter in range(max_iters):
        i = iter % len(training_pairs) 
        true_y, example = training_pairs[i]
        weights = new_weights_for_one_example(example, weights, true_y)


        if (iter % 1000) == 0:
            _, acc = get_accuracy(my_training_pairs, weights)
            print("iteration {}, training accuracy: {}"
                  .format(iter, acc))
    return weights

def batch_SGD(training_pairs, initial_weights, max_passes=100):
    """Build up a gradient for all the training pairs, apply it, and then do it
    again and again.
    Returns a new set of weights."""

    weights = np.copy(initial_weights)
    my_training_pairs = copy.copy(training_pairs)
    random.shuffle(my_training_pairs)

    best_acc = 0
    best_weights = None
    total_gradient = np.zeros(weights.size)
    for iter in range(max_passes):
        for true_y, example in training_pairs:
            gradient = gradient_for_one_example(example, weights, true_y)
            total_gradient += gradient

        weights = weights - LEARNING_RATE * (total_gradient / len(training_pairs))
        _, acc = get_accuracy(my_training_pairs, weights)
        print("batch {}, training accuracy: {}" .format(iter, acc))
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
    return best_weights

def prob_positive(example, weights):
    """Takes example and weights, both of which should be numpy arrays.

    Returns a PROBABILITY that this is a positive example.
    """
    return sigmoid(example.dot(weights))

def get_accuracy(pairs, trained_weights):
    ncorrect = 0
    for (true_c, example) in pairs:
        estimate = prob_positive(example, trained_weights)
        guess = round(estimate)
        if guess == true_c:
            ncorrect += 1
    return (ncorrect, ncorrect / len(pairs))

def main():
    data_init()
    document_pairs = load_movie_documents()
    # document_pairs = classifier_util.load_binary_toy_documents()
    training, test = classifier_util.split_training_test(document_pairs)

    print("got {} training instances, {} test instances"
          .format(len(training), len(test)))

    V = extract_vocabulary(training)
    indices_to_words = {V[word] : word for word in V}

    training_pairs = []
    for (cl, tokens) in training:
        example = tokens_to_example(tokens, V)
        training_pairs.append((1 if cl == "pos" else 0, example))

    testing_pairs = []
    for (cl, tokens) in test:
        example = tokens_to_example(tokens, V)
        testing_pairs.append((1 if cl == "pos" else 0, example))

    # you could do randomized initial weights, like this
    initial_weights = np.random.default_rng().standard_normal(len(V) + 1)
    # or just zeros, it doesn't seem to matter much
    # initial_weights = np.zeros(len(V) + 1)
    trained_weights = batch_SGD(training_pairs, initial_weights,
                                 max_passes=200)
    print(trained_weights)
    ncorrect, accuracy = get_accuracy(testing_pairs, trained_weights)

    print("FINAL ACCURACY: {} / {} = {:0.2f}".format(
          ncorrect, len(test), accuracy))

    if False:
        print()
        print("some words with high weights")
        most_salient = np.argpartition(trained_weights, -50)[-50:]
        for index in most_salient:
            print(indices_to_words[index], trained_weights[index])

        print()
        print("and some words with low weights")
        most_salient = np.argpartition(trained_weights, 50)[:50]
        for index in most_salient:
            print(indices_to_words[index], trained_weights[index])

if __name__ == "__main__": main()
