#!/usr/bin/env python3

"""
Quick demonstration of how to load up the movie reviews corpus from NLTK, for
sentiment analysis purposes.

Also shows some simple NLTK sentence segmentation and tokenization strategies.
"""

import nltk

from nltk.corpus import movie_reviews

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/movie_reviews")
except LookupError:
    nltk.download("movie_reviews")
    nltk.download("punkt")

for fileid in nltk.corpus.movie_reviews.fileids():
    # get the raw text
    text = movie_reviews.raw(fileid)

    category = "pos" if fileid.startswith("pos/") else "neg"

    # get all the sentences
    for sentence in nltk.sent_tokenize(text):
        tokens = nltk.word_tokenize(sentence)
        # now we have a list of tokens in "tokens"
        print(category, tokens[:5], ". . .")

