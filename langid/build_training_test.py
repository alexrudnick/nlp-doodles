#!/usr/bin/env python3

import os
import random
from pathlib import Path

def main():
    examples = []

    for lang in "en es fr it de".split():
        with open(Path.home() / "langid" / "sentences.{}.txt".format(lang)) as infile:
            for line in infile:
                text = line.strip()
                examples.append((lang, text))

    
    random.seed(42069)

    random.shuffle(examples)
    testsize = len(examples) // 10
    testset = examples[:testsize]
    trainingset = examples[testsize:]

    with open("testset.tsv", "w") as outfile:
        for lang, text in testset:
            print("{}\t{}".format(lang, text), file=outfile)

    with open("trainingset.tsv", "w") as outfile:
        for lang, text in trainingset:
            print("{}\t{}".format(lang, text), file=outfile)

if __name__ == "__main__": main()
