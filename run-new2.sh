#!/bin/bash
# New runs in 2023. This time attempting to make the models work properly for speaker independent
# setups

stage=0
seed=2602

. cmd.sh
. path.sh
. utils/parse_options.sh

set -eu


if [ $stage -le 16 ]; then
  local/chain/build_new_tree.sh \
    --num_leaves 350 \
    --traindata data/uit-sme-segmented-and-giellagas-train/ \
    --trainali exp/uit-sme-segmented-and-giellagas-train/tri3a_ali_uit-sme-segmented-and-giellagas-train \
    --validali exp/uit-sme-segmented-and-giellagas-train/tri3a_ali_giellagas-valid \
    exp/chain/tree2-simple2
fi

num_units=$(tree-info exp/chain/tree2-simple2/tree | grep "num-pdfs" | cut -d" " -f2)

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh \
    --treedir exp/chain/tree2-simple2 \
    --sharddir "shards/uit-sme-segmented-and-giellagas-train-sub2-simple2"
  local/chain/prep-shards.sh \
    --shardname "ali.valid.JOB.gz" \
    --datadir data/giellagas-valid/ \
    --treedir exp/chain/tree2-simple2 \
    --sharddir "shards/giellagas-valid-sub2-simple2"
fi

if [ $stage -le 18 ]; then
  local/chain/prepare_graph_clustered.sh \
    --graph exp/chain/graph2-simple2 \
    --trainset uit-sme-segmented-and-giellagas-train \
    --treedir exp/chain/tree2-simple2
fi

if [ $stage -le 19 ]; then
  local/chain/run_training.sh \
    --treedir exp/chain/tree2-simple2 \
    --py_script local/chain/sb-train-mtl-w2v2.py \
    --hparams "hyperparams/mtl/New-w2w2-F3-E-simple2.yaml" 
  local/chain/run_training.sh \
    --treedir exp/chain/tree2-simple2 \
    --py_script local/chain/sb-train-mtl-w2v2.py \
    --hparams "hyperparams/mtl/New-w2w2-F3-E-simple2-main.yaml" 
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_bpe.1000.varikn/ exp/chain/graph2-simple2 exp/chain/graph2-simple2/graph_bpe.1000.varikn
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh \
    --tree exp/chain/tree2-simple2 \
    --datadir "data/giellagas-valid/" \
    --py_script "local/chain/sb-test-w2v2-mtl-avg.py" \
    --hparams "hyperparams/mtl/New-w2w2-F3-E-simple2-main.yaml" \
    --graphdir exp/chain/graph2-simple2/graph_bpe.1000.varikn \
    --decodedir "exp/chain/New-2023/Main-$seed-${num_units}units-E-simple2/decode_giellagas_valid"
fi


