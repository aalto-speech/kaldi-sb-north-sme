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
        action="append", default=["<UNK>", "<SENT>"])
args = parser.parse_args()

nonsilence = set()
with open(args.nonsilence) as fi:
    for line in fi:
        phone = line.strip()
        nonsilence |= set((phone,))

with open(args.textfile) as fi:
    for line in fi:
        for word in line.strip().split():
            if word in args.specials:
                pass
            elif not nonsilence.issuperset(word):
                print("FOUND OFFENDING LINE!:")
                print(line)
                for char in word:
                    if char not in nonsilence:
                        print("THIS CHAR:", char)
                # Not good!
                sys.exit(1)

# All good
sys.exit(0)

