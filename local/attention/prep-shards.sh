#!/bin/bash

cmd="srun --mem 8G --time 4:0:0 -c 2"
datadir="data/uit-sme-segmented-and-giellagas-train/"
sharddir="shards/train-sub2"
nj=2

. path.sh
. parse_options.sh

if [ ! -d $datadir/split$nj ]; then
  utils/split_data.sh $datadir $nj
fi

$cmd local/chain/make_shards.py $nj "$sharddir" \
  --num-proc 0 \
  --segments "$datadir"/split$nj/JOB/segments \
             "$datadir"/split$nj/JOB/wav.scp \
  --text     "$datadir"/split$nj/JOB/text

