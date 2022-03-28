#!/bin/bash
sb_cmd="srun -c 2 --mem 24G --time 1-12:0:0 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink"
hyperparams="hyperparams/attention/w2v2-non-indep.yaml"

. path.sh
. utils/parse_options.sh
 
$sb_cmd python local/attention/train-w2v2.py $hyperparams --test_data_id "test"
