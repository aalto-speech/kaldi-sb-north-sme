#!/bin/bash
set -eu

mode="strict"

. utils/parse_options.sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <reftext> <hyptext>"
  exit 1
fi

reftext="$1"
hyptext="$2"


local/wer_ref_filter <"$reftext" > "$reftext".filt
local/character_tokenizer.py <"$reftext".filt > "$reftext".filt.chars
local/character_tokenizer.py <"$hyptext" > "$hyptext".chars

module load kaldi-strawberry
echo "WER:"
compute-wer --mode="$mode" ark:"$reftext".filt ark:"$hyptext"
echo "CER:"
compute-wer --mode="$mode" ark:"$reftext".filt.chars ark:"$hyptext".chars
