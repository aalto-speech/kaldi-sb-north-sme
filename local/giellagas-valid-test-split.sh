#!/bin/bash
set -eu

# Some of the speakers should count as training data:
train_spks="IV_f3 IV_m3 IV_m5 sN_Fs_Ed_fl sN_Ss_Ed_m1"

# Use some of the interviewers as validation data:
valid_spks="IV_f1 IV_f2 IV_m1 IV_m2"

mkdir data/giellagas-train
for spk in $train_spks; do
  echo "$spk" >> data/giellagas-train/spklist
done

utils/subset_data_dir_tr_cv.sh --cv-spk-list data/giellagas-train/spklist data/giellagas data/giellagas-tmp data/giellagas-train

utils/combine_data.sh data/uit-sme-segmented-and-giellagas-train data/uit-sme-segmented/ data/giellagas-train

mkdir data/giellagas-valid
for spk in $valid_spks; do
  echo "$spk" >> data/giellagas-valid/spklist
done

utils/subset_data_dir_tr_cv.sh --cv-spk-list data/giellagas-valid/spklist data/giellagas-tmp data/giellagas-test data/giellagas-valid

rm -rf data/giellagas-tmp
