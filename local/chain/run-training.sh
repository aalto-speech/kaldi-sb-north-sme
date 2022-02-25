#!/bin/bash

cmd="srun --mem 24G --time 2-0:0 -c2 --gres=gpu:1 --constraint volta -p dgx-spa,dgx-common,gpu,gpu-nvlink"

. path.sh
. parse_options.sh

$cmd python local/chain/sb-train-mtl-am-w2v2.py hyperparams/mtl/w2v2-A.yaml
