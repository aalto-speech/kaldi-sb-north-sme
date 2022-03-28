#!/bin/bash

. path.sh
. parse_options.sh

mkdir -p data/non-indep/lm
cut -f2- -d" " data/non-indep/train/text | sed -E "s/<UNK> ?//g" > data/non-indep/lm/train.plain
cut -f2- -d" " data/non-indep/valid/text | sed -E "s/<UNK> ?//g" > data/non-indep/lm/valid.plain

sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/non-indep/lm/train.plain \
  >data/non-indep/lm/train.boundaries
sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/non-indep/lm/valid.plain \
  >data/non-indep/lm/valid.boundaries

local/lm/train_lm.sh --lmdatadir data/non-indep/lm --dict_dir data/non-indep/local/dict_bpe400 --expdir exp/non-indep/lm --BPE_units 400 data/non-indep/lang_transcript_bpe400
