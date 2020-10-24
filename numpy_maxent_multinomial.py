#!/usr/bin/env python3

"""
Another pass at doing classification, like numpy_maxent_classifier, except now
we're treating this as a multinomial classifier, with softmax.
"""

import copy
import random
import nltk
import sys
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

def gradient_for_one_example(example, weights, y):
    """
    Given an example vector and a weights vector and the true class index for
    that example, return the gradient matrix for that example.

    This is the partial derivative of the loss function with respect to each
    particular weight parameter. So the output vector will be the same size as
    the weight matrix, ie, n_classes x example size.


    derivative of L_CE wrt weight k
    = -( 1{y = k} - p(y=k|x)) * x_k

    What do they mean by x_k here? Is it the specific features for class k? I
    think it's going to just be the example...
    """
    dist = softmax(weights.dot(example))
    gradient = np.zeros(weights.shape)

    # these are matrices here -- they were vectors in the binary case.
    l2_penalty = (2 * LAMBDA_2 * weights)
    l1_penalty = (LAMBDA_1 * np.sign(weights))

    # XXX: egregiously not array style
    for k in range(0, dist.size):
        left = 1 if y == k else 0
        gradient_for_k = -1 * (left - dist[k]) * example
        gradient[k] = gradient_for_k
    return gradient + l2_penalty + l1_penalty

def get_accuracy(pairs, trained_weights):
    ncorrect = 0
    for (true_c, example) in pairs:
        dist = softmax(trained_weights.dot(example))
        max_class = np.argmax(dist)
        if max_class == true_c:
            ncorrect += 1
    return (ncorrect, ncorrect / len(pairs))

def batch_SGD(training_pairs, initial_weights, max_passes=1000):
    """Build up a gradient for all the training pairs, apply it, and then do it
    again and again.
    Returns a new set of weights."""

    weights = np.copy(initial_weights)
    my_training_pairs = copy.copy(training_pairs)
    random.shuffle(my_training_pairs)

    best_acc = float("-inf") 
    best_weights = None
    total_gradient = np.zeros(weights.shape)
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
        if acc == 1.0:
            print("completely fit the training set; bailing!")
            return best_weights
    return best_weights

def softmax(vector):
    """Returns a vector of the same size as the input vector."""
    # clip values at 100 to avoid overflows. MATHS.
    vector = np.minimum(vector, 100)
    total = sum(np.exp(vector))
    return np.exp(vector) / total

def classname_to_index(clname, classes):
    if clname in classes:
        return classes.index(clname)
    else:
        message = "class {} not in the seen classes??".format(clname)
        raise ValueError(message)

def document_classification_demo(document_pairs):
    training, test = classifier_util.split_training_test(document_pairs)

    # sorted list of classes.
    classes = list(sorted(set(c for c,_ in training)))

    V = extract_vocabulary(training)
    indices_to_words = {V[word] : word for word in V}

    # first dimension is the class index
    # second index is the feature index
    initial_weights = np.random.default_rng().standard_normal([len(classes),
                                                              len(V) + 1])

    training_pairs = []
    for (cl, tokens) in training:
        example = tokens_to_example(tokens, V)
        training_pairs.append((classname_to_index(cl, classes), example))

    testing_pairs = []
    for (cl, tokens) in test:
        example = tokens_to_example(tokens, V)
        testing_pairs.append((classname_to_index(cl, classes), example))

    print("initial!")
    print(initial_weights)
    trained_weights = batch_SGD(training_pairs, initial_weights,
                                max_passes=200)
    print()
    print("trained!")
    print(trained_weights)

    ncorrect, accuracy = get_accuracy(testing_pairs, trained_weights)
    print("FINAL ACCURACY: {} / {} = {:0.2f}".format(
          ncorrect, len(test), accuracy))

def iris_train_demo():
    iris_data = classifier_util.load_iris_example_pairs()
    train_pairs, test_pairs = classifier_util.split_training_test(iris_data)

    n_classes = 3
    # 4 normal features with a bias term too.
    n_features = 5

    initial_weights = np.random.default_rng().standard_normal([n_classes,
                                                               n_features])
    trained_weights = batch_SGD(train_pairs, initial_weights, max_passes=2000)
    print()
    print("trained!")
    print(trained_weights)
    ncorrect, accuracy = get_accuracy(test_pairs, trained_weights)
    print("FINAL ACCURACY: {} / {} = {:0.2f}".format(
          ncorrect, len(test_pairs), accuracy))

def main():
    data_init()

    if len(sys.argv) > 1 and sys.argv[1] == "toy":
        document_pairs = classifier_util.load_multinomial_toy_documents()
        document_classification_demo(document_pairs)
    elif len(sys.argv) > 1 and sys.argv[1] == "iris":
        iris_train_demo()
    else:
        document_pairs = load_movie_documents()
        document_classification_demo(document_pairs)

    # to cheat you can use these weights to solve the toy example exactly.
    # weights = np.array([[-10, 1, 1, 0],
    #                     [1, -10, 1, 0],
    #                     [1, 1, -10, 0]])

if __name__ == "__main__": main()
