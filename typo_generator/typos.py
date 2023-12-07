#!/usr/bin/env/python3

import fileinput
import random

TYPO_WEIGHTS = {
        "qwerty_row": 0.05,
        "qwerty_column": 0.1,
        "case_swap": 0.1,
        "transpositions": 0.05,
        "homophones": 0.5,
}

QWERTY = ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]

# shoutouts to lkuper for more homophones
HOMOPHONES = [
        ["their", "they're", "there"],
        ["you're", "your"],
        ["two", "to", "too"],
        ["write", "right", "wright"],
        ["break", "brake"],
        ["coarse", "course"],
        ["by", "bye", "buy"],
        ["effect", "affect"]
]

def case_swap(text: str) -> str:
    new_chars = []
    for c in text:
        new_char = c
        if random.random() < TYPO_WEIGHTS["case_swap"]:
            if(c.islower()):
                new_char = c.upper()
            elif(c.isupper()):
                new_char = c.lower()
        new_chars.append(new_char)
    return "".join(new_chars)

def same_row_for(c):
    for row in QWERTY:
        if c.lower() in row:
            index = row.find(c.lower())
            if index == 0:
                newindex = index + 1
            elif index == len(row) - 1:
                newindex = index - 1
            else:
                newindex = index + random.choice([1, -1])

            if c.islower():
                return row[newindex]
            else:
                return row[newindex].upper()
    return c

def qwerty_misinputs(text: str) -> str:
    new_chars = []
    for c in text:
        new_char = c
        if random.random() < TYPO_WEIGHTS["qwerty_row"]:
            new_char = same_row_for(c)
        new_chars.append(new_char)
    return "".join(new_chars)

def transpositions(text: str) -> str:
    characters = list(text)

    for i in range(1, len(characters)):
        if random.random() < TYPO_WEIGHTS["transpositions"]:
            c = characters[i]
            characters[i] = characters[i - 1]
            characters[i - 1] = c
    return "".join(characters)

def homophones(text: str) -> str:
    tokens = text.split()
    out = []
    for token in tokens:
        newtoken = token
        for row in HOMOPHONES:
            if token in row and random.random() < TYPO_WEIGHTS["homophones"]:
                newrow = row[:]
                newrow.remove(token)
                newtoken = random.choice(newrow)
        out.append(newtoken)
    return " ".join(out)

def add_typos(text: str) -> str:
    out = text
    for funk in [homophones, qwerty_misinputs, case_swap, transpositions]:
        out = funk(out)
    return out

def main():
    for line in fileinput.input():
        print(add_typos(line.strip()))

if __name__ == "__main__": main()
