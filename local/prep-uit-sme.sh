#!/bin/bash

CORPUS_ROOT="/teamwork/t40511_asr/c/uit-sme"

. path.sh
. parse_options.sh

mkdir -p data/uit-sme

if [ -e data/uit-sme/utt2spk ]; then
  echo "data/uit-sme/utt2spk already exists, not overwriting"
  exit 0 
fi

export LC_ALL="fi_FI.utf8"  # Must handle Sami characters in text
for wavfile in $CORPUS_ROOT/16k/*/*.wav; do
  fname=$(basename $wavfile .wav)
  spk=$(basename $(dirname $wavfile))
  uttid=$spk-$fname
  echo "$uttid $spk" >> data/uit-sme/utt2spk
  echo "$uttid $wavfile" >> data/uit-sme/wav.scp
  echo -n "$uttid " >> data/uit-sme/text
  local/preprocess_uitsme.py <$CORPUS_ROOT/txt/$spk/$fname.txt >> data/uit-sme/text
done
export LC_ALL=C

utils/fix_data_dir.sh data/uit-sme
