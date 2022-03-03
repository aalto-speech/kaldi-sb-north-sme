#!/bin/bash
set -eu

. ./cmd.sh

maxorder=10 #-n --norder
cutoffs="0 0 1" #-O --cutoffs
scale=0.001 #-D --dscale
prune_scale= #-E --dscale2
extra_args="--clear_history --3nzer --arpa" #-C -3 -a
cmd="$train_cmd"

. ./path.sh
. parse_options.sh

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <datacmd> <devdatacmd> <out-dir>"
  echo " Data command should output data to stdout"
  echo " E.G. cat data/lm/train.txt |"
  exit 1
fi

data="$1"
devdata="$2"
out=$3

echo "Training VariKN N-gram model"

if [ -z "$prune_scale" ]; then
  prune_scale=$(echo "$scale * 2" | bc) # 2 x scale is the recommendation
fi

#Sentence boundary tags:
train_data_str="$data | sed -e 's:^:<s> :' -e 's:$: </s>:' |"
dev_data_str="$devdata | sed -e 's:^:<s> :' -e 's:$: </s>:' |"


$cmd $out/log/varigram_kn.log \
  varigram_kn \
  --cutoffs="$cutoffs" \
  --dscale=$scale \
  --dscale2=$prune_scale \
  --norder=$maxorder \
  $extra_args \
  --opti "$dev_data_str" \
  "$train_data_str" $out/varikn.lm.gz
