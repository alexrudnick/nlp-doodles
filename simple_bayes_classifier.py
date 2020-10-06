#!/usr/bin/env python3

"""
Toy implementation of a Naive Bayes classifier, based on Jurafsky & Martin SLP,
Figure 4.2 from https://web.stanford.edu/~jurafsky/slp3/4.pdf
"""

import nltk
from math import log2
from collections import defaultdict
from collections import Counter
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
        # get all the sentences
        for sentence in nltk.sent_tokenize(text):
            tokens = nltk.word_tokenize(sentence)
            # now we have a list of tokens in "tokens"
            document.extend(tokens)
        pairs.append((category, document))
    return pairs

def train_naive_bayes(D, C):
    """
    D is the list of documents, pairs of the form (cat, [token, ...])
    C is the collection of possible classes

    returns log priors P(c), log likelihoods P(w | c), and vocabulary V
    """
    V = set()
    N_doc = len(D)

    logpriors = defaultdict(float)
    loglikelihoods = defaultdict(float)

    for (cat, tokens) in D:
        # naively including every word in the vocabulary!
        # This could be trimmed down, or we could do some preprocessing, maybe
        # ignore rare words.
        V.update(tokens)

    for cat in C:
        N_c = len([c for (c, _) in D if c == cat])
        logpriors[cat] = log2(N_c / N_doc)
        bigdoc = []
        count = Counter()
        for c,tokens in D:
            if c == cat:
                bigdoc.extend(tokens)
        print("bigdoc has {} tokens for category {}".format(len(bigdoc), cat))
        count.update(bigdoc)
        print("(the,{}) = {}".format(cat, count["the"]))

        # no need to compute this in the loop, will be the same every time
        denominator = sum((count[wprime] + 1) for wprime in V)
        for wordtype in V:
            numerator = count[wordtype] + 1
            loglikelihoods[(wordtype, cat)] = log2(numerator/denominator)
    return logpriors, loglikelihoods, V

def test_naive_bayes(testdoc, logpriors, loglikelihoods, C, V):
    """Given testdoc, a list of tokens, return the most likely class."""
    print("TESTING: {}".format(" ".join(testdoc[:10])))
    bestscore = float("-inf")
    # http://hrwiki.org/w/index.php?title=Flagrant_System_Error&redirect=no
    bestclass = "Flagrant System Error"
    for c in C:
        sum = logpriors[c]
        for tok in testdoc:
            if tok in V:
                sum += loglikelihoods[(tok, c)]

        print("score for {}: {}".format(c, sum))
        if sum > bestscore:
            bestscore = sum
            bestclass = c
    return bestclass

def main():
    data_init()
    document_pairs = load_movie_documents()
    classes = set(c for c,_ in document_pairs)

    logpriors, loglikelihoods, V = train_naive_bayes(document_pairs, classes)

    guess = test_naive_bayes("boring no fun uninspired long".split(),
                             logpriors, loglikelihoods, classes, V)
    print(guess)
    guess = test_naive_bayes("amazing exciting fun".split(),
                             logpriors, loglikelihoods, classes, V)
    print(guess)


if __name__ == "__main__": main()
