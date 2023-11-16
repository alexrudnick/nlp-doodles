#!/usr/bin/env python3

import nltk

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from collections import Counter

def extract_features_chr_trigrams(text):
    feature_dict = Counter()
    for (a, b, c) in zip(text, text[1:], text[2:]):
        feature_dict["{}{}{}".format(a, b, c)] += 1
    return feature_dict

def load_dataset(filename):
    out = []
    with open(filename) as infile:
        for line in infile:
            try:
                cl, text = line.strip().split("\t")
                featuredict = extract_features_chr_trigrams(text)
                out.append(((featuredict, cl), text))
            except:
                continue
    return out

def main():
    THETOL = 1e-4

    classifier = SklearnClassifier(LogisticRegression(C=1,
                                   penalty='l2',
                                   tol=THETOL))

    trainingset = load_dataset("trainingset.tsv")
    testset = load_dataset("trainingset.tsv")

    classifier.train([example for (example, text) in trainingset])

    correct = 0
    incorrect = 0
    for ((features, cl), text) in testset:
        predicted = classifier.classify(features)
        if(predicted == cl):
            correct += 1
        else:
            incorrect += 1
            print("INCORRECT!! should be {}, was {}".format(cl, predicted))
            print("  the text: {}".format(text))
    print("correct: {}, incorrect: {}".format(correct, incorrect))

if __name__ == "__main__": main()
