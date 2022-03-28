#!/bin/bash

. path.sh
. parse_options.sh

mkdir -p data/giellagas-only/lm
cut -f2- -d" " data/giellagas-only/train/text | sed -E "s/<UNK> ?//g" > data/giellagas-only/lm/train.plain
cut -f2- -d" " data/giellagas-only/valid/text | sed -E "s/<UNK> ?//g" > data/giellagas-only/lm/valid.plain

sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/giellagas-only/lm/train.plain \
  >data/giellagas-only/lm/train.boundaries
sed -e "s/^/<s> /g" -e "s/$/ <s>/g" \
  <data/giellagas-only/lm/valid.plain \
  >data/giellagas-only/lm/valid.boundaries

local/lm/train_lm.sh --lmdatadir data/giellagas-only/lm --dict_dir data/giellagas-only/local/dict_bpe400 --expdir exp/giellagas-only/lm --BPE_units 400 data/giellagas-only/lang_transcript_bpe400
