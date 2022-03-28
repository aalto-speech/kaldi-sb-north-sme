#!/bin/bash


# For legacy!
stage=8
index=1  #Split index!

. cmd.sh
. utils/parse_options.sh

if [ $stage -le 8 ]; then
  local/loo-cv/transcript-lm.sh --index $index
fi

if [ $stage -le 9 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/loo-cv/$index/train/
  steps/compute_cmvn_stats.sh data/loo-cv/$index/train
fi

if [ $stage -le 10 ]; then
  steps/train_mono.sh --nj 2 --cmd "$train_cmd" \
    data/loo-cv/$index/train data/lang_train/ exp/loo-cv/$index/mono1a || exit 1;
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/mono1a exp/loo-cv/$index/mono1a_ali_train || exit 1;

  steps/train_deltas.sh --cmd "$train_cmd" 1200 5000 \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/mono1a_ali_train exp/loo-cv/$index/tri1a || exit 1;
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/tri1a exp/loo-cv/$index/tri1a_ali_train || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 1200 5000 \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/tri1a_ali_train exp/loo-cv/$index/tri2a || exit 1;
fi

if [ $stage -le 13 ]; then
  steps/align_si.sh  --nj 2 --cmd "$train_cmd" \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/tri2a/ exp/loo-cv/$index/tri2a_ali_train || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 1200 5000 \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/tri2a_ali_train exp/loo-cv/$index/tri3a || exit 1;
fi

################################################################################
### Stage 14-15 Align uit-sme-segmented and valid, Create new lang and tree
################################################################################


if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" \
    data/loo-cv/$index/train data/lang_train exp/loo-cv/$index/tri3a exp/loo-cv/$index/tri3a_ali_train || exit 1;
  # ALI NOT FROM HERE:
  #steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" --retry-beam 100 \
  #  data/giellagas-valid/ data/lang_train exp/loo-cv/$index/tri3a exp/loo-cv/$index/tri3a_ali_giellagas-valid || exit 1;
  # WAS PRODUCED WITH THIS COMMAND:
  # steps/align_fmllr.sh --nj 2 --cmd "slurm.pl --mem 4G --time 1:0:0" data/giellagas-valid data/lang_train exp/giellagas-and-uit-sme/tri3a exp/loo-cv/external_ali_giellagas-valid
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
    --traindata data/loo-cv/$index/train/ \
    --trainali exp/loo-cv/$index/tri3a_ali_train \
    --validali exp/loo-cv/external_ali_giellagas-valid/ \
    exp/loo-cv/$index/chain/tree2
fi

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh \
    --treedir exp/loo-cv/$index/chain/tree2 \
    --datadir data/loo-cv/$index/train \
    --sharddir shards/loo-cv/$index/train 
  local/chain/prep-shards.sh \
    --treedir exp/loo-cv/$index/chain/tree2 \
    --datadir data/loo-cv/$index/valid \
    --sharddir shards/loo-cv/$index/valid \
    --shardname "ali.valid.JOB.gz"
fi

if [ $stage -le 18 ]; then
  local/chain/prepare_graph_clustered.sh \
    --dataroot data/loo-cv/$index/ \
    --trainset train \
    --validset valid \
    --treedir exp/loo-cv/$index/chain/tree2 \
    --graph exp/loo-cv/$index/chain/graph2
fi

if [ $stage -le 19 ]; then
  local/chain/run-training.sh \
    --treedir exp/loo-cv/$index/chain/tree2 \
    --hparams "hyperparams/mtl/w2v2-loo-cv.yaml --loo_cv_index '$index'"
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bpe.1000.varikn/ exp/loo-cv/$index/chain/graph2/ exp/loo-cv/$index/chain/graph2/graph_bpe.1000.varikn
  utils/mkgraph.sh --self-loop-scale 1.0 data/loo-cv/$index/lang_transcript_bpe400 exp/loo-cv/$index/chain/graph2/ exp/loo-cv/$index/chain/graph2/graph_transcript_bpe400
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-loo-cv.yaml --loo_cv_index '$index'" \
    --datadir data/loo-cv/$index/test \
    --tree exp/loo-cv/$index/chain/tree2 \
    --graphdir exp/loo-cv/$index/chain/graph2/graph_bpe.1000.varikn \
    --decodedir exp/loo-cv/$index/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test$index \
    --acwt 1.0 --post-decode-acwt 10.0
fi

if [ $stage -le 22 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/mtl/w2v2-loo-cv.yaml --loo_cv_index '$index'" \
    --datadir data/loo-cv/$index/test \
    --tree exp/loo-cv/$index/chain/tree2 \
    --graphdir exp/loo-cv/$index/chain/graph2/graph_transcript_bpe400 \
    --decodedir exp/loo-cv/$index/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test$index-transcriptlm \
    --stage 2 --posteriors_from exp/loo-cv/$index/chain/sb-mtl-am/wav2vec2-sub2-F/decode-test$index \
    --acwt 1.0 --post-decode-acwt 10.0
fi
