#!/bin/bash

cmd="srun --mem 8G --time 4:0:0 -c 3"
treedir="exp/chain/tree2/"
shard_affix="-sub2"

. path.sh
. parse_options.sh

$cmd local/chain/make_shards.py 2 shards/uit-sme-segmented${shard_name} \
  --num-proc 2 \
  --segments data/uit-sme-segmented/split2/JOB/segments \
             data/uit-sme-segmented/split2/JOB/wav.scp \
  --aliark   "gunzip -c $treedir/ali.JOB.gz | ali-to-pdf $treedir/final.mdl ark:- ark:- |"

$cmd local/chain/make_shards.py 2 shards/giellagas-valid${shard_name} \
  --num-proc 2 \
  --segments data/giellagas-valid/split2/JOB/segments \
             data/giellagas-valid/split2/JOB/wav.scp \
  --aliark   "gunzip -c $treedir/ali.valid.JOB.gz | ali-to-pdf $treedir/final.mdl ark:- ark:- |"

