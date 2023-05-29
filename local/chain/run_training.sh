#!/bin/bash

cmd="srun --mem 20G --time 15:0:0 -c2 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink"
hparams="hyperparams/chain/New-CRDNN-J.yaml"
treedir="exp/chain/tree/"
py_script="local/chain/sb-train-mtl.py"

. path.sh
. parse_options.sh

num_units=$(tree-info $treedir/tree | grep "num-pdfs" | cut -d" " -f2)

timesfailed=0
while ! $cmd python $py_script $hparams --num_units $num_units; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 20 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

