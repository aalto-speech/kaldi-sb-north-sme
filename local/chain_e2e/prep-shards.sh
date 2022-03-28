#!/bin/bash

cmd="srun --mem 8G --time 4:0:0 -c 2"
datadir="data/uit-sme-segmented-and-giellagas-train/"
sharddir="shards/train-sub2"

. path.sh
. parse_options.sh

$cmd local/chain/make_shards.py 2 "$sharddir" \
  --num-proc 0 \
  --segments "$datadir"/split2/JOB/segments \
             "$datadir"/split2/JOB/wav.scp 

