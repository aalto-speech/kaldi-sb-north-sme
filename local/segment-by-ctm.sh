#!/bin/bash

other_opts=

. ./path.sh
. parse_options.sh

if [ "$#" != 3 ]; then
  echo "Usage: $0 <in-data> <ctmfile> <out-data>"
  exit 1
fi

set -eu

indir="$1"
ctm="$2"
outdir="$3"

rec2dur=
if [ -e "$indir"/utt2dur ]; then
  rec2dur="--rec2dur $indir/utt2dur"
fi

mkdir -p "$outdir"
cp "$indir"/wav.scp "$outdir"/wav.scp
export LC_ALL="fi_FI.utf8"  # Must handle Sami characters in text
local/ctm-to-segments.py $rec2dur $other_opts "$ctm" "$indir"/utt2spk "$outdir"

export LC_ALL=C
sort "$outdir"/utt2spk -o "$outdir"/utt2spk
utils/fix_data_dir.sh "$outdir"
