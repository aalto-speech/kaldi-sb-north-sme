#!/bin/bash

traintexts="/m/teamwork/t40511_asr/p/sami/lmdata/sme/tromso.txt /m/teamwork/t40511_asr/p/sami/lmdata/sme/wikipedia_full.txt"
validtext="data/giellagas-valid/text"
nonsilence_phones="data/local/uit-sme-dict/nonsilence_phones.txt"

. path.sh
. parse_options.sh

set -eu

mkdir -p data/lm

rm -f data/lm/train.boundaries
for textf in $traintexts; do
  local/lm/preprocess-to-sents.py "$nonsilence_phones" "$textf" >> data/lm/train.boundaries
done
sed -e "s:<s> ::g" -e "s: </s>::g" data/lm/train.boundaries > data/lm/train.plain

cut -f2- -d " " $validtext > data/lm/valid.boundaries
sed -e "s:<s> ::g" -e "s: </s>::g" data/lm/valid.boundaries > data/lm/valid.plain
