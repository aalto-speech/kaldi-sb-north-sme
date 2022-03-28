#!/bin/bash
stage=0

. cmd.sh
. utils/parse_options.sh
 
# Make the LOO-CV splits elsewhere (local/loo-cv/make-split.sh)

if [ $stage -le 0 ]; then
  local/attention/prep-shards.sh --datadir data/giellagas-only/valid --sharddir shards/giellagas-only/attention-valid
fi

if [ $stage -le 1 ]; then
  local/attention/prep-shards.sh --datadir data/giellagas-only/train --sharddir shards/giellagas-only/attention-train
fi

if [ $stage -le 2 ]; then
  local/attention/prep-shards.sh --datadir data/giellagas-only/test --sharddir shards/giellagas-only/attention-test
fi

if [ $stage -le 3 ]; then
  local/attention/run-training.sh --hyperparams hyperparams/attention/w2v2-giellagas-only.yaml
fi

if [ $stage -le 4 ]; then
  local/attention/test.sh --hyperparams hyperparams/attention/w2v2-giellagas-only.yaml
fi
