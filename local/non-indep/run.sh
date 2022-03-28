#!/bin/bash


# For legacy!
stage=9

. cmd.sh
. utils/parse_options.sh

if [ $stage -le 9 ]; then
  local/non-indep/transcript-lm.sh
fi

if [ $stage -le 10 ]; then
  steps/train_mono.sh --nj 2 --cmd "$train_cmd" \
    data/non-indep/train data/lang_train/ exp/non-indep/mono1a || exit 1;
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/non-indep/train data/lang_train exp/non-indep/mono1a exp/non-indep/mono1a_ali_train || exit 1;

  steps/train_deltas.sh --cmd "$train_cmd" 1200 5000 \
    data/non-indep/train data/lang_train exp/non-indep/mono1a_ali_train exp/non-indep/tri1a || exit 1;
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/non-indep/train data/lang_train exp/non-indep/tri1a exp/non-indep/tri1a_ali_train || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 1200 5000 \
    data/non-indep/train data/lang_train exp/non-indep/tri1a_ali_train exp/non-indep/tri2a || exit 1;
fi

if [ $stage -le 13 ]; then
  steps/align_si.sh  --nj 2 --cmd "$train_cmd" \
    data/non-indep/train data/lang_train exp/non-indep/tri2a/ exp/non-indep/tri2a_ali_train || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 1200 5000 \
    data/non-indep/train data/lang_train exp/non-indep/tri2a_ali_train exp/non-indep/tri3a || exit 1;
fi

################################################################################
### Stage 14-15 Align uit-sme-segmented and valid, Create new lang and tree
################################################################################


if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" \
    data/non-indep/train data/lang_train exp/non-indep/tri3a exp/non-indep/tri3a_ali_train || exit 1;
  steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" --retry-beam 100 \
    data/non-indep/valid/ data/lang_train exp/non-indep/tri3a exp/non-indep/tri3a_ali_valid || exit 1;
fi

if [ $stage -le 15 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang_train/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang_train..."
      echo " ... not sure what to do. Exiting."
      exit 1;
    fi
  else
    cp -r data/lang_train data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 16 ]; then
  local/chain/build_new_tree.sh \
    --traindata data/non-indep/train/ \
    --trainali exp/non-indep/tri3a_ali_train \
    --validali exp/non-indep/tri3a_ali_valid \
    exp/non-indep/chain/tree2
fi

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh \
    --treedir exp/non-indep/chain/tree2 \
    --datadir data/non-indep/train \
    --sharddir shards/non-indep/train 
  local/chain/prep-shards.sh \
    --treedir exp/non-indep/chain/tree2 \
    --datadir data/non-indep/valid \
    --sharddir shards/non-indep/valid \
    --shardname "ali.valid.JOB.gz"
fi

if [ $stage -le 18 ]; then
  local/chain/prepare_graph_clustered.sh \
    --dataroot data/non-indep/ \
    --trainset train \
    --validset valid \
    --treedir exp/non-indep/chain/tree2 \
    --graph exp/non-indep/chain/graph2
fi

if [ $stage -le 19 ]; then
  local/chain/run-training.sh \
    --treedir exp/non-indep/chain/tree2 \
    --hparams "hyperparams/mtl/w2v2-non-indep.yaml"
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bpe.1000.varikn/ exp/non-indep/chain/graph2/ exp/non-indep/chain/graph2/graph_bpe.1000.varikn
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-non-indep.yaml" \
    --datadir data/non-indep/test \
    --tree exp/non-indep/chain/tree2 \
    --graphdir exp/non-indep/chain/graph2/graph_bpe.1000.varikn \
    --decodedir exp/non-indep/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test
fi

if [ $stage -le 22 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/non-indep/lang_transcript_bpe400/ exp/non-indep/chain/graph2/ exp/non-indep/chain/graph2/graph_transcript_bpe400
fi

if [ $stage -le 23 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-non-indep.yaml" \
    --datadir data/non-indep/valid/ \
    --tree exp/non-indep/chain/tree2 \
    --graphdir exp/non-indep/chain/graph2/graph_transcript_bpe400 \
    --decodedir exp/non-indep/chain/sb-mtl-am/wav2vec2-sub2-F/decode-valid-transcript \
    --acwt 1.0 --post-decode-acwt 10.0 
fi

if [ $stage -le 24 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-non-indep.yaml" \
    --datadir data/non-indep/test/ \
    --tree exp/non-indep/chain/tree2 \
    --graphdir exp/non-indep/chain/graph2/graph_transcript_bpe400 \
    --decodedir exp/non-indep/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test-transcript \
    --acwt 1.0 --post-decode-acwt 10.0 
fi
