#!/bin/bash

cmd="srun --mem 24G --time 2-0:0 -c2 --gres=gpu:1 --constraint volta -p dgx-spa"
hparams="hyperparams/mtl/w2v2-C.yaml"
treedir="exp/chain/tree2/"
py_script="local/chain_e2e/sb-train-lfmmi-am-w2v2.py"

. path.sh
. parse_options.sh

num_units=$(tree-info $treedir/tree | grep "num-pdfs" | cut -d" " -f2)

while ! $cmd python $py_script $hparams --num_units $num_units; do
  echo "Training crashed, restarting!"
done
