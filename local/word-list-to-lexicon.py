#!/usr/bin/env python3

import fileinput
import argparse

parser = argparse.ArgumentParser(
    "Take a word list and spit out a lexicon.txt (character phone units)"
)
parser.add_argument("input", help="Path to file or - for stdin")
parser.add_argument("--specials", help="Special tokens to ignore (add them yourself)", 
        action="append", default=["<UNK>", "<SENT>"])
args = parser.parse_args()

for line in fileinput.input(args.input, openhook=fileinput.hook_encoded("utf-8")):
    word = line.strip()
    if not word:
        continue
    elif word in args.specials:
        continue
    else:
        print(word, " ".join(word))

