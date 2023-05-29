#!/bin/bash
# New runs in 2023. This time attempting to make the models work properly for speaker independent
# setups

stage=0
seed=2602

. cmd.sh
. path.sh
. utils/parse_options.sh

set -eu


num_units=$(tree-info exp/chain/tree2/tree | grep "num-pdfs" | cut -d" " -f2)


if [ $stage -le 19 ]; then
  local/chain/run_training.sh \
    --treedir exp/chain/tree2 \
    --py_script local/chain/sb-train-mtl-w2v2.py \
    --hparams "hyperparams/mtl/New-w2w2-F3-F.yaml" 
  local/chain/run_training.sh \
    --treedir exp/chain/tree2 \
    --py_script local/chain/sb-train-mtl-w2v2.py \
    --hparams "hyperparams/mtl/New-w2w2-F3-F-main.yaml" 
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh \
    --tree exp/chain/tree2 \
    --datadir "data/giellagas-valid/" \
    --py_script "local/chain/sb-test-w2v2-mtl-avg.py" \
    --hparams "hyperparams/mtl/New-w2w2-F3-E-simple-main.yaml" \
    --graphdir exp/chain/graph2/graph_bpe.1000.varikn \
    --decodedir "exp/chain/New-2023/Main-$seed-${num_units}units-E-simple/decode_giellagas_valid"
fi


