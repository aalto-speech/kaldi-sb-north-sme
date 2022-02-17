#!/usr/bin/env bash
# Recipe built around two sami language corpora:
# uit-sme (training, dev)
# giellagas (testing, bootstrap)
#
# Both are Northern-Sami corpora
# Giellagas should function as an adequate test set: it
# has a decent number of different speakers and hours
# However, uit-sme only has very long audio recordings
# So, to bootstrap the recipe, we will first train some initial models
# with Giellagas (which has short segments)
# and then use those models to align uit-sme
# And then split uit-sme into smaller segments
# And then finally we will models on the segmented uit-sme
# and test on giellagas!

stage=0

. cmd.sh
. utils/parse_options.sh

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

if [ $stage -le 2 ]; then
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/giellagas data/lang_train/ exp/giellagas/mono1a || exit 1;
fi
