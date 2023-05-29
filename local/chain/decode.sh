#!/bin/bash
# Script to test 

am_cmd="srun --gres=gpu:1 --constraint=volta --time 0:20:0 --mem 20G -p dgx-common,dgx-spa,gpu,gpushort"
decode_cmd="slurm.pl --mem 24G --time 4:0:0"
score_cmd="slurm.pl --mem 2G --time 0:30:0"
nj=6
hparams="hyperparams/chain/New-CRDNN-J.yaml"
datadir="data/dev_clean/"
decodedir="exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean"
tree="exp/chain/tree/"
graphdir="exp/chain/graph/graph_bpe.5000.varikn"
py_script="local/chain/sb-test-mtl-avg.py"
posteriors_from=
score_opts=

# Decoding params:
acwt=1.0
post_decode_acwt=10.0
beam=15
lattice_beam=8

# Script stage
stage=0
skip_scoring="false"

. path.sh
. parse_options.sh

set -eu

posteriors_prefix="$decodedir/logprobs"
mkdir -p $decodedir

if [ $stage -le 1 ]; then
  num_units=$(tree-info "$tree"/tree | grep "num-pdfs" | cut -d" " -f2)
  test_in="--testdir $datadir"
  $am_cmd python $py_script $hparams --num_units $num_units \
    $test_in \
    --test_probs_out "$posteriors_prefix".from_sb
  # Make SCPs:
  copy-matrix ark:"$posteriors_prefix".from_sb ark,scp:"$posteriors_prefix".ark,"$posteriors_prefix".scp
fi

# Lattice generation
if [ $stage -le 2 ]; then 
  if [ -d $posteriors_from ]; then
    ln -s -f "$PWD"/"$posteriors_from"/logprobs* "$decodedir"/
  fi
  utils/split_scp.pl "$posteriors_prefix".scp $(for n in `seq $nj`; do echo "$posteriors_prefix"."$n".scp; done)
  $decode_cmd JOB=1:$nj "$decodedir"/log/decode.JOB.log \
    latgen-faster-mapped \
    --acoustic-scale=$acwt \
    --beam=$beam \
    --lattice_beam=$lattice_beam \
    --word-symbol-table="$graphdir/words.txt" \
    $tree/final.mdl \
    "$graphdir/HCLG.fst" \
    scp:"$posteriors_prefix".JOB.scp \
    "ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c > $decodedir/lat.JOB.gz"
  echo "$nj" > "$decodedir"/num_jobs
fi

if [[ $stage -le 3 && ! $skip_scoring == "true" ]]; then
  local/score.sh \
    $score_opts \
    --cmd "$score_cmd" \
    --beam $lattice_beam \
    "$datadir" \
    "$graphdir" \
    "$decodedir"
  steps/scoring/score_kaldi_cer.sh \
    $score_opts \
    --stage 2 \
    --cmd "$score_cmd" \
    --beam $lattice_beam \
    "$datadir" \
    "$graphdir" \
    "$decodedir"
fi
