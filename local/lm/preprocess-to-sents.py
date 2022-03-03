#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser(
        """Check that a text file has graphemes associated with the given nonsilence phones list
        """
)
parser.add_argument("nonsilence", help="The Kaldi dict style list of nonsilence phones")
parser.add_argument("textfile", help="The text file to check")
parser.add_argument("--specials", help="Special tokens, to be ignored", 
        action="append", default=["<UNK>", "<SENT>", "<s>", "</s>"])
args = parser.parse_args()

nonsilence = set()
with open(args.nonsilence) as fi:
    for line in fi:
        phone = line.strip()
        nonsilence |= set((phone,))

with open(args.textfile) as fi:
    for line in fi:
        filtered = []
        for word in line.strip().split():
            if word in args.specials:
                pass
            elif not nonsilence.issuperset(word):
                if filtered:
                    print(" ".join(filtered))
                filtered = []
            else:
                filtered.append(word)
        if filtered:
            print(" ".join(filtered))

