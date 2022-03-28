#!/bin/bash

. path.sh
. parse_options.sh

mkdir -p data/transcript-lm
cut -f2- -d" " data/uit-sme-segmented-and-giellagas-train/text | sed -E "s/<UNK> ?//g" > data/transcript-lm/train.plain
cut -f2- -d" " data/giellagas-valid/text | sed -E "s/<UNK> ?//g" > data/transcript-lm/valid.plain

sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/transcript-lm/train.plain \
  >data/transcript-lm/train.boundaries
sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/transcript-lm/valid.plain \
  >data/transcript-lm/valid.boundaries

local/lm/train_lm.sh --lmdatadir data/transcript-lm --dict_dir data/local/dict_transcript_bpe400 --expdir exp/transcript-lm --BPE_units 400 data/lang_transcript_bpe400
