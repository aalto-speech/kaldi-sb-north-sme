#!/bin/bash

stage=1
index=1  #Split index!

. cmd.sh
. utils/parse_options.sh
 
# Make the LOO-CV splits elsewhere (local/loo-cv/make-split.sh)

if [ $stage -le 0 ]; then
  local/attention/prep-shards.sh --datadir data/loo-cv/$index/valid --sharddir shards/loo-cv/attention-valid
fi

if [ $stage -le 1 ]; then
  local/attention/prep-shards.sh --datadir data/loo-cv/$index/train --sharddir shards/loo-cv/$index/attention-train
fi

if [ $stage -le 2 ]; then
  local/attention/run-training.sh --index $index
fi

if [ $stage -le 3 ]; then
  local/attention/prep-shards.sh --nj 1 --datadir data/loo-cv/$index/test --sharddir shards/loo-cv/$index/attention-test
fi

if [ $stage -le 4 ]; then
  local/attention/test.sh --index $index
fi
