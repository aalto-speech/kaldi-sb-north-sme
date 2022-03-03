#!/usr/bin/env bash
# Recipe built around two sami language corpora:
# uit-sme (training, dev)
# giellagas (testing, bootstrap)
#
# Both are Northern-Sami corpora
# Giellagas should function as an adequate test set: it
# has 19 speakers and ~2 hours in total
# However, uit-sme only has very long audio recordings
# So, to bootstrap the recipe, we will first train some initial models
# with Giellagas (which has short segments)
#
# and then use those models to align uit-sme and build a combined data model
# And then split uit-sme into smaller segments
# And then finally we will build models on the segmented uit-sme
# and test on giellagas!

stage=0

. cmd.sh
. utils/parse_options.sh


################################################################################
### Stages 0-1 Data preparation
################################################################################

if [ $stage -le 0 ]; then
  local/prep-giellagas.sh
  local/prep-uit-sme.sh
  local/check_grapheme_sets_compatible.py data/giellagas/text data/uit-sme/text
  local/check_grapheme_sets_compatible.py data/uit-sme/text data/giellagas/text 
  local/prepare_lexicon.sh --extra_texts "data/giellagas/text" data/uit-sme data/local/uit-sme-dict data/lang_train
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/giellagas
  steps/compute_cmvn_stats.sh data/giellagas
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/uit-sme
  steps/compute_cmvn_stats.sh data/uit-sme
fi


################################################################################
### Stages 2-5 Train models on Giellagas
################################################################################

if [ $stage -le 2 ]; then
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/giellagas data/lang_train/ exp/giellagas/mono1a || exit 1;
fi

if [ $stage -le 3 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/giellagas data/lang_train exp/giellagas/mono1a exp/giellagas/mono1a_ali_giellagas || exit 1;

  steps/train_deltas.sh --cmd "$train_cmd" 1200 5000 \
    data/giellagas data/lang_train exp/giellagas/mono1a_ali_giellagas exp/giellagas/tri1a || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/giellagas data/lang_train exp/giellagas/tri1a exp/giellagas/tri1a_ali_giellagas || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 1200 5000 \
    data/giellagas data/lang_train exp/giellagas/tri1a_ali_giellagas exp/giellagas/tri2a || exit 1;
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
    data/giellagas data/lang_train exp/giellagas/tri2a/ exp/giellagas/tri2a_ali_giellagas || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 1200 5000 \
    data/giellagas data/lang_train exp/giellagas/tri2a_ali_giellagas exp/giellagas/tri3a || exit 1;
fi

################################################################################
### Stage 6 Align all data, build combined model
################################################################################

if [ $stage -le 6 ]; then
  utils/combine_data.sh data/giellagas-and-uit-sme data/giellagas data/uit-sme
  # Very large beam to allow alignment to work
  steps/align_fmllr.sh --nj 12 --beam 40 --retry_beam 140 --cmd "$train_cmd" \
    data/giellagas-and-uit-sme data/lang_train exp/giellagas/tri3a exp/giellagas/tri3a_ali_giellagas-and-uit-sme
  steps/train_sat.sh --cmd "$train_cmd" \
    --retry_beam 80 \
    1200 7500 \
    data/giellagas-and-uit-sme \
    data/lang_train \
    exp/giellagas/tri3a_ali_giellagas-and-uit-sme \
    exp/giellagas-and-uit-sme/tri3a || exit 1;
fi

################################################################################
### Stage 7 Align uit-sme with combined model, segment, create giellagas split
################################################################################

if [ $stage -le 7 ]; then
  steps/align_fmllr.sh --nj 2 --beam 40 --retry_beam 100 --cmd "$train_cmd" \
    data/uit-sme data/lang_train exp/giellagas-and-uit-sme/tri3a exp/giellagas-and-uit-sme/tri3a_ali_uit-sme
fi

if [ $stage -le 8 ]; then
  steps/get_train_ctm.sh data/uit-sme data/lang_train exp/giellagas-and-uit-sme/tri3a_ali_uit-sme
  local/segment-by-ctm.sh data/uit-sme exp/giellagas-and-uit-sme/tri3a_ali_uit-sme/ctm data/uit-sme-segmented
  local/giellagas-valid-test-split.sh
fi

################################################################################
### Stage 9-13 Train with uit-sme-segmented from scratch
################################################################################

if [ $stage -le 9 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 2 data/uit-sme-segmented
  steps/compute_cmvn_stats.sh data/uit-sme-segmented
fi

if [ $stage -le 10 ]; then
  steps/train_mono.sh --nj 2 --cmd "$train_cmd" \
    data/uit-sme-segmented data/lang_train/ exp/uit-sme-segmented/mono1a || exit 1;
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/mono1a exp/uit-sme-segmented/mono1a_ali_uit-sme-segmented || exit 1;

  steps/train_deltas.sh --cmd "$train_cmd" 1200 5000 \
    data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/mono1a_ali_uit-sme-segmented exp/uit-sme-segmented/tri1a || exit 1;
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj 2 --cmd "$train_cmd" \
    data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/tri1a exp/uit-sme-segmented/tri1a_ali_uit-sme-segmented || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 1200 5000 \
    data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/tri1a_ali_uit-sme-segmented exp/uit-sme-segmented/tri2a || exit 1;
fi

if [ $stage -le 13 ]; then
  steps/align_si.sh  --nj 2 --cmd "$train_cmd" \
    data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/tri2a/ exp/uit-sme-segmented/tri2a_ali_uit-sme-segmented || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" 1200 5000 \
    data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/tri2a_ali_uit-sme-segmented exp/uit-sme-segmented/tri3a || exit 1;
fi

################################################################################
### Stage 14-15 Align uit-sme-segmented and valid, Create new lang and tree
################################################################################


if [ $stage -le 14 ]; then
  #steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" \
  #  data/uit-sme-segmented data/lang_train exp/uit-sme-segmented/tri3a exp/uit-sme-segmented/tri3a_ali_uit-sme-segmented || exit 1;
  steps/align_fmllr.sh --nj 2 --cmd "$train_cmd" --retry-beam 100 \
    data/giellagas-valid/ data/lang_train exp/giellagas/tri3a exp/giellagas/tri3a_ali_giellagas-valid || exit 1;
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
  local/chain/build_new_tree.sh exp/chain/tree2
fi

if [ $stage -le 17 ]; then
  local/chain/prep-shards.sh
fi

if [ $stage -le 18 ]; then
  local/chain/prepare_graph_clustered.sh
fi

if [ $stage -le 19 ]; then
  local/chain/run-training.sh
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh data/lang_bpe.1000.varikn/ exp/chain/graph2/ exp/chain/graph2/graph_bpe.1000.varikn
fi

if [ $stage -le 21 ]; then
  local/chain/decode.sh 
fi

