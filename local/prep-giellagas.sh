#!/bin/bash
set -eu

CORPUS_ROOT="/teamwork/t40511_asr/scratch/rouhea1/Giellagas/giellagas-north/"

. path.sh
. parse_options.sh

mkdir -p data/giellagas

if [ -e data/giellagas/utt2spk ]; then
  echo "data/giellagas/utt2spk already exists, not overwriting"
  exit 0 
fi

export LC_ALL="fi_FI.utf8"  # Must handle Sami characters in text
for wavfile in "$CORPUS_ROOT"/*/*/*.wav; do
  recid=$(basename $wavfile .wav)
  dialect=$(basename $(dirname $(dirname $wavfile)))
  echo "$recid sox -t wav $wavfile -t wav -L -b 16 -r 16000 - remix - |" >> data/giellagas/wav.scp
  local/parse-giellagas.py $CORPUS_ROOT/$dialect/Annotations/${recid}.eaf $recid data/giellagas
done
export LC_ALL=C

sort data/giellagas/utt2spk -o data/giellagas/utt2spk
utils/fix_data_dir.sh data/giellagas
