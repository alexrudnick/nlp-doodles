#!/usr/bin/env python3

import pronouncing
import nltk
import readline

initial = [
    "raindrops on roses and whiskers on kittens",
    "bright copper kettles and warm woolen mittens",
    "brown paper packages tied up with strings",
    "these are a few of my favorite things",
    ]

def stress_for_sentence(words: list[str]):
    stresses = []
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if len(phones) == 0:
            print("unknown word", word)
            return ""
        stress = pronouncing.stresses(phones[0])

        stresses.append(stress)

    out = "".join(stresses)
    out = out.replace("2", "1")
    return out

def can_have_same_stress(line1, line2):
    ## XXX -- always assume that we take the first pronunciation for each word.
    stress1 = stress_for_sentence(line1)
    stress2 = stress_for_sentence(line2)

    print(" current stress:", stress1)
    print("proposed stress:", stress2)
    return stress1 == stress2
    # return True

def print_verse(verse: list[list[str]]):
    for lineno, words in enumerate(verse):
        print("{}: {}".format(lineno+1, " ".join(words)))

def main():
    verse = [nltk.word_tokenize(text) for text in initial]

    print_verse(verse)
    while True:
        command = ""
        try:
            command = input("> ")
        except EOFError as e:
            print()
            break
        command = command.lower()
        if command == "help" or command == "?":
            print("editing commands are: help, ?, replace <line>, list, exit")
        elif command == "exit":
            print()
            break
        elif command == "list":
            print_verse(verse)
        elif command.startswith("replace "):
            _, num = command.split()
            num = int(num) - 1
            if num not in range(len(verse)):
                print("unknown line number")
                continue
            print("please enter a line with the same meter as: ",
                    " ".join(verse[num]))
            newtext = input("> ")
            newline = nltk.word_tokenize(newtext)
            if can_have_same_stress(verse[num], newline):
                verse[num] = newline
            else:
                print("meter mismatch; y to replace anyway")
                anyway = input("> ")
                if anyway.lower() == "y":
                    verse[num] = newline
        else:
            print("?")

if __name__ == "__main__": main()
