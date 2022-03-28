#!/bin/bash


# For legacy!
stage=9
index=1  #Split index!

. cmd.sh
. utils/parse_options.sh


if [ $stage -le 10 ]; then
  steps/train_mono.sh --nj 2 --cmd "$train_cmd" \
    data/giellagas-only/train data/lang_train/ exp/giellagas-only/mono1a || exit 1;
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/giellagas-only/train data/lang_train exp/giellagas-only/mono1a exp/giellagas-only/mono1a_ali_train || exit 1;

  steps/train_deltas.sh --cmd "$train_cmd" 1200 5000 \
    data/giellagas-only/train data/lang_train exp/giellagas-only/mono1a_ali_train exp/giellagas-only/tri1a || exit 1;
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/giellagas-only/train data/lang_train exp/giellagas-only/tri1a exp/giellagas-only/tri1a_ali_train || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 1200 5000 \
    data/giellagas-only/train data/lang_train exp/giellagas-only/tri1a_ali_train exp/giellagas-only/tri2a || exit 1;
fi

if [ $stage -le 13 ]; then
  steps/align_si.sh  --nj 2 --cmd "$train_cmd" \
    data/giellagas-only/train data/lang_train exp/giellagas-only/tri2a/ exp/giellagas-only/tri2a_ali_train || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 1200 5000 \
    data/giellagas-only/train data/lang_train exp/giellagas-only/tri2a_ali_train exp/giellagas-only/tri3a || exit 1;
fi

################################################################################
### Stage 14-15 Align uit-sme-segmented and valid, Create new lang and tree
################################################################################


if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" \
    data/giellagas-only/train data/lang_train exp/giellagas-only/tri3a exp/giellagas-only/tri3a_ali_train || exit 1;
  steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" --retry-beam 100 \
    data/giellagas-only/valid/ data/lang_train exp/giellagas-only/tri3a exp/giellagas-only/tri3a_ali_valid || exit 1;
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
    --traindata data/giellagas-only/train/ \
    --trainali exp/giellagas-only/tri3a_ali_train \
    --validali exp/giellagas-only/tri3a_ali_valid \
    exp/giellagas-only/chain/tree2
fi

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh \
    --treedir exp/giellagas-only/chain/tree2 \
    --datadir data/giellagas-only/train \
    --sharddir shards/giellagas-only/train 
  local/chain/prep-shards.sh \
    --treedir exp/giellagas-only/chain/tree2 \
    --datadir data/giellagas-only/valid \
    --sharddir shards/giellagas-only/valid \
    --shardname "ali.valid.JOB.gz"
fi

if [ $stage -le 18 ]; then
  local/chain/prepare_graph_clustered.sh \
    --dataroot data/giellagas-only/ \
    --trainset train \
    --validset valid \
    --treedir exp/giellagas-only/chain/tree2 \
    --graph exp/giellagas-only/chain/graph2
fi

if [ $stage -le 19 ]; then
  local/chain/run-training.sh \
    --treedir exp/giellagas-only/chain/tree2 \
    --hparams "hyperparams/mtl/w2v2-giellagas-only.yaml"
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bpe.1000.varikn/ exp/giellagas-only/chain/graph2/ exp/giellagas-only/chain/graph2/graph_bpe.1000.varikn
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-giellagas-only.yaml" \
    --datadir data/giellagas-only/test \
    --tree exp/giellagas-only/chain/tree2 \
    --graphdir exp/giellagas-only/chain/graph2/graph_bpe.1000.varikn \
    --decodedir exp/giellagas-only/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test$index-2
fi

if [ $stage -le 22 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/giellagas-only/lang_transcript_bpe400/ exp/giellagas-only/chain/graph2/ exp/giellagas-only/chain/graph2/graph_transcript_bpe400
fi

if [ $stage -le 23 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-giellagas-only.yaml" \
    --datadir data/giellagas-only/valid/ \
    --tree exp/giellagas-only/chain/tree2 \
    --graphdir exp/giellagas-only/chain/graph2/graph_transcript_bpe400 \
    --decodedir exp/giellagas-only/chain/sb-mtl-am/wav2vec2-sub2-F/decode-valid-transcript \
    --acwt 1.0 --post-decode-acwt 10.0 
fi

if [ $stage -le 24 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-giellagas-only.yaml" \
    --datadir data/giellagas-only/test/ \
    --tree exp/giellagas-only/chain/tree2 \
    --graphdir exp/giellagas-only/chain/graph2/graph_transcript_bpe400 \
    --decodedir exp/giellagas-only/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test-transcript \
    --acwt 1.0 --post-decode-acwt 10.0 
fi
