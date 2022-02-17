#!/usr/bin/env python3
import sys
import re

sentence_bound = re.compile(r"[.!?]")
boundary = " <SENT> "  # Note spaces here
unk_token = "<UNK>"

def get_lines():
    for line in sys.stdin:
        line = line.strip()
        for sentence in sentence_bound.split(line):
            yield sentence

lines = []
for line in get_lines():
    line = line.replace(",", "")
    if not line:
        continue
    line = "".join(
            char if char.isalpha() else " " 
            for char in line
    )
    # This could process initialisms
    # but there also seem to be some ALL CAPS words
    #line = " ".join(
    #        " ".join(word) if word.isupper() else word
    #        for word in line.split()
    #)
    line = line.lower()
    line = " ".join(
            unk_token if any(c in word for c in "wåäöæø") else word 
            for word in line.split()
    )
    lines.append(line)

print(boundary.join(lines))

