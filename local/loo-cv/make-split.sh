#!/bin/bash

# The available training / LOO speakers:
spks="IV_f3 IV_m3 IV_m5 X_m1 sN_Fs_Ed_fl sN_Fs_Ed_m1 sN_Fs_Wd_f1 sN_Fs_Wd_m1 sN_Ss_Ed_f1 sN_Ss_Ed_m1 sN_Tsd_Cs_m1 sN_Tsd_Fwd_f1 sN_Tsd_Fwd_m1 sN_Tsd_Gad_f1 sN_Tsd_Gad_m1"

# Use some of the interviewers as validation data:
valid_spks="IV_f1 IV_f2 IV_m1 IV_m2"

mkdir -p data/loo-cv/
index=1
for spk in $spks; do
  mkdir -p data/loo-cv/$index/test
  echo "$spk" > data/loo-cv/$index/test/spklist
  utils/subset_data_dir_tr_cv.sh \
    --cv-spk-list data/loo-cv/$index/test/spklist \
    data/giellagas \
    data/loo-cv/$index/giellagas-train \
    data/loo-cv/$index/test 
  utils/combine_data.sh data/loo-cv/$index/train data/uit-sme-segmented/ data/loo-cv/$index/giellagas-train
  utils/copy_data_dir.sh data/giellagas-valid/ data/loo-cv/$index/valid
  index=$((index+1))
done

# NOTE: The validation split is not here currently.

# TENTATIVELY: Keep splits 5,6,7,8,10,11,12,14,15
# 1 IV_f3 
# 2 IV_m3 
# 3 IV_m5 
# 4 X_m1 
# 5 sN_Fs_Ed_fl 
# 6 sN_Fs_Ed_m1 
# 7 sN_Fs_Wd_f1 
# 8 sN_Fs_Wd_m1 
# 9 sN_Ss_Ed_f1 
# 10 sN_Ss_Ed_m1 
# 11 sN_Tsd_Cs_m1 
# 12 sN_Tsd_Fwd_f1 
# 13 sN_Tsd_Fwd_m1 
# 14 sN_Tsd_Gad_f1 
# 15 sN_Tsd_Gad_m1
