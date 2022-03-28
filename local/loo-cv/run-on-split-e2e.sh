#!/bin/bash

stage=3
index=1  #Split index!

. cmd.sh
. utils/parse_options.sh

if [ $stage -le 0 ]; then
  local/loo-cv/transcript-lm.sh --index $index
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/loo-cv/$index/train/
  steps/compute_cmvn_stats.sh data/loo-cv/$index/train
fi

if [ $stage -le 2 ]; then
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

if [ $stage -le 3 ]; then
  local/chain_e2e/build_new_tree.sh \
    --type biphone \
    --min_biphone_count 50 \
    --min_monophone_count 10 \
    --tie true \
    data/loo-cv/$index/train/ \
    data/lang_chain \
    exp/loo-cv/$index/chain_e2e/tree2
fi

if [ $stage -le 4 ]; then
  if [ ! -d shards/loo-cv/$index/train ]; then
    local/chain_e2e/prep-shards.sh \
      --datadir data/loo-cv/$index/train \
      --sharddir shards/loo-cv/$index/train 
  fi
fi

if [ $stage -le 5 ]; then
  local/chain/prepare_graph_clustered.sh \
    --dataroot data/loo-cv/$index/ \
    --trainset train \
    --validset valid \
    --treedir exp/loo-cv/$index/chain_e2e/tree2 \
    --graph exp/loo-cv/$index/chain_e2e/graph2
fi

if [ $stage -le 6 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/loo-cv/$index/chain_e2e/tree2 \
    --hparams "hyperparams/mtl/w2v2-loo-cv-e2e.yaml --loo_cv_index '$index'"
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bpe.1000.varikn/ exp/loo-cv/$index/chain_e2e/graph2/ exp/loo-cv/$index/chain_e2e/graph2/graph_bpe.1000.varikn
  utils/mkgraph.sh --self-loop-scale 1.0 data/loo-cv/$index/lang_transcript_bpe400 exp/loo-cv/$index/chain_e2e/graph2/ exp/loo-cv/$index/chain_e2e/graph2/graph_transcript_bpe400
fi

if [ $stage -le 8 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-loo-cv-e2e.yaml --loo_cv_index '$index'" \
    --datadir data/loo-cv/$index/test \
    --tree exp/loo-cv/$index/chain_e2e/tree2 \
    --graphdir exp/loo-cv/$index/chain_e2e/graph2/graph_bpe.1000.varikn \
    --decodedir exp/loo-cv/$index/chain_e2e/sb-mtl-am/wav2vec2-sub2-F/decode-test$index \
    --py_script "local/chain/sb-test-lfmmi-w2v2.py" \
    --acwt 1.0 --post-decode-acwt 10.0
fi

if [ $stage -le 9 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-loo-cv-e2e.yaml --loo_cv_index '$index'" \
    --datadir data/loo-cv/$index/test \
    --tree exp/loo-cv/$index/chain_e2e/tree2 \
    --graphdir exp/loo-cv/$index/chain_e2e/graph2/graph_transcript_bpe400 \
    --decodedir exp/loo-cv/$index/chain_e2e/sb-mtl-am/wav2vec2-sub2-F/decode-test$index-transcriptlm \
    --py_script "local/chain/sb-test-lfmmi-w2v2.py" \
    --stage 2 --posteriors_from exp/loo-cv/$index/chain_e2e/sb-mtl-am/wav2vec2-sub2-F/decode-test$index \
    --acwt 1.0 --post-decode-acwt 10.0
fi
