#!/bin/bash

traintexts="/m/teamwork/t40511_asr/p/sami/lmdata/sme/tromso.txt /m/teamwork/t40511_asr/p/sami/lmdata/sme/wikipedia_full.txt"
validtext="data/giellagas-valid/text"
nonsilence_phones="data/local/uit-sme-dict/nonsilence_phones.txt"
lmdatadir="data/lm"

. path.sh
. parse_options.sh

set -eu

mkdir -p $lmdatadir

rm -f $lmdatadir/train.boundaries
for textf in $traintexts; do
  local/lm/preprocess-to-sents.py "$nonsilence_phones" "$textf" >> $lmdatadir/train.boundaries
done
sed -e "s:<s> ::g" -e "s: </s>::g" $lmdatadir/train.boundaries > $lmdatadir/train.plain

cut -f2- -d " " $validtext > $lmdatadir/valid.boundaries
sed -e "s:<s> ::g" -e "s: </s>::g" $lmdatadir/valid.boundaries > $lmdatadir/valid.plain
