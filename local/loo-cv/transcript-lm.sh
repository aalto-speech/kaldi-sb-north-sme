#!/bin/bash

index=5

. path.sh
. parse_options.sh

mkdir -p data/loo-cv/$index/lm
cut -f2- -d" " data/loo-cv/$index/train/text | sed -E "s/<UNK> ?//g" > data/loo-cv/$index/lm/train.plain
cut -f2- -d" " data/loo-cv/$index/valid/text | sed -E "s/<UNK> ?//g" > data/loo-cv/$index/lm/valid.plain

sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/loo-cv/$index/lm/train.plain \
  >data/loo-cv/$index/lm/train.boundaries
sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/loo-cv/$index/lm/valid.plain \
  >data/loo-cv/$index/lm/valid.boundaries

local/lm/train_lm.sh --lmdatadir data/loo-cv/$index/lm --dict_dir data/loo-cv/$index/local/dict_bpe400 --expdir exp/loo-cv/$index/lm --BPE_units 400 data/loo-cv/$index/lang_transcript_bpe400

