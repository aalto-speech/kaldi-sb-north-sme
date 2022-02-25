#!/bin/bash
set -eu

# Use some of the interviewers as validation data:
valid_spks="IV_f1 IV_f2 IV_m1 IV_m2"

mkdir data/giellagas-valid
for spk in $valid_spks; do
  echo "$spk" >> data/giellagas-valid/spklist
done

utils/subset_data_dir_tr_cv.sh --cv-spk-list data/giellagas-valid/spklist data/giellagas data/giellagas-test data/giellagas-valid
