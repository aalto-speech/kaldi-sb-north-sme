#!/usr/bin/env python3

import sys
for line in sys.stdin:
    uttid, *words = line.strip().split()
    chars = []
    for word in words:
        chars.extend(word)
    print(uttid, " ".join(chars))
