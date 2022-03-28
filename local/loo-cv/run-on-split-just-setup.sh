#!/bin/bash


# For legacy!
stage=8
index=1  #Split index!

. cmd.sh
. utils/parse_options.sh

if [ $stage -le 8 ]; then
  local/loo-cv/transcript-lm.sh --index $index
fi

if [ $stage -le 9 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/loo-cv/$index/train/
  steps/compute_cmvn_stats.sh data/loo-cv/$index/train
fi

if [ $stage -le 16 ]; then
  local/chain/build_new_tree.sh \
    --traindata data/loo-cv/$index/train/ \
    --trainali exp/loo-cv/$index/tri3a_ali_train \
    --validali exp/loo-cv/external_ali_giellagas-valid/ \
    exp/loo-cv/$index/chain/tree2
fi

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh \
    --treedir exp/loo-cv/$index/chain/tree2 \
    --datadir data/loo-cv/$index/train \
    --sharddir shards/loo-cv/$index/train 
fi
