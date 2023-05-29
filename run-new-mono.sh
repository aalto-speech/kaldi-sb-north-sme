#!/bin/bash
# New runs in 2023. This time attempting to make the models work properly for speaker independent
# setups

stage=0
seed=2602

. cmd.sh
. utils/parse_options.sh

set -eu


if [ $stage -le 16 ]; then
  local/chain_e2e/build_new_tree.sh \
    data/uit-sme-segmented-and-giellagas-train/ \
    data/lang_chain/ \
    exp/chain/tree2-mono

  trainali=exp/uit-sme-segmented-and-giellagas-train/tri3a_ali_uit-sme-segmented-and-giellagas-train 
  validali=exp/uit-sme-segmented-and-giellagas-train/tri3a_ali_giellagas-valid 
  treedir=exp/chain/tree2-mono
  frame_subsampling_factor=2

  # Train
  echo "Converting Train alignments"
  nj=$(cat ${trainali}/num_jobs)
  $train_cmd JOB=1:$nj ${treedir}/log/convert.train.JOB \
    convert-ali --frame-subsampling-factor=$frame_subsampling_factor \
      ${trainali}/final.mdl ${treedir}/1.mdl ${treedir}/tree \
      "ark:gunzip -c $trainali/ali.JOB.gz|" "ark:|gzip -c >$treedir/ali.JOB.gz" || exit 1;

  echo "Converting validation alignments"
  nj=$(cat ${validali}/num_jobs)
  $train_cmd JOB=1:$nj ${treedir}/log/convert.valid.JOB \
    convert-ali --frame-subsampling-factor=$frame_subsampling_factor \
      ${validali}/final.mdl ${treedir}/1.mdl ${treedir}/tree \
      "ark:gunzip -c $validali/ali.JOB.gz|" "ark:|gzip -c >$treedir/ali.valid.JOB.gz" || exit 1;
fi

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh \
    --treedir exp/chain/tree2-mono \
    --sharddir "shards/uit-sme-segmented-and-giellagas-train-sub2-mono"
  local/chain/prep-shards.sh \
    --shardname "ali.valid.JOB.gz" \
    --datadir data/giellagas-valid/ \
    --treedir exp/chain/tree2-mono \
    --sharddir "shards/giellagas-valid-sub2-mono"
fi

if [ $stage -le 18 ]; then
  local/chain/prepare_graph_clustered.sh \
    --graphdir exp/chain/graph2-mono \
    --trainset uit-sme-segmented-and-giellagas-train \
    --treedir exp/chain/tree2-mono
fi

if [ $stage -le 19 ]; then
  local/chain/run_training.sh \
    --treedir exp/chain/tree2-mono \
    --py_script local/chain/sb-train-mtl-w2v2.py \
    --hparams hyperparams/mtl/New-w2w2-F3.yaml
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_bpe.1000.varikn/ exp/chain/graph2-mono exp/chain/graph2-mono/graph_bpe.1000.varikn
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/New-w2w2-F3-E-mono.yaml" \
    --decodedir "exp/chain/New-2023/Initial-<seed>-<num_units>units-E-mono"
fi

