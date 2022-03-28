#!/bin/bash
stage=0

. cmd.sh
. utils/parse_options.sh
 
# Make the LOO-CV splits elsewhere (local/loo-cv/make-split.sh)

if [ $stage -le 0 ]; then
  local/attention/prep-shards.sh --datadir data/giellagas-valid --sharddir shards/unseen/attention-valid
fi

if [ $stage -le 1 ]; then
  local/attention/prep-shards.sh --datadir data/uit-sme-segmented-and-giellagas-train --sharddir shards/unseen/attention-train
fi

if [ $stage -le 2 ]; then
  local/attention/prep-shards.sh --datadir data/giellagas-test/ --sharddir shards/unseen/attention-test
fi

if [ $stage -le 3 ]; then
  local/attention/run-training.sh --hyperparams hyperparams/attention/w2v2-unseen.yaml
fi

if [ $stage -le 4 ]; then
  local/attention/test.sh --hyperparams hyperparams/attention/w2v2-unseen.yaml
fi
