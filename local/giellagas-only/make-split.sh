#!/bin/bash
set -eu

# The available speakers:
spks="sN_Fs_Ed_fl sN_Fs_Wd_f1 sN_Fs_Wd_m1 sN_Ss_Ed_f1 sN_Ss_Ed_m1 sN_Tsd_Cs_m1 sN_Tsd_Fwd_f1 sN_Tsd_Fwd_m1 sN_Tsd_Gad_f1 sN_Tsd_Gad_m1"
lowuttspks="IV_f3 IV_m3 IV_m5 X_m1 IV_f1 IV_f2 IV_m1 IV_m2"

mkdir -p data/giellagas-only/staff

for spk in $lowuttspks; do
  echo "$spk" >> data/giellagas-only/staff/spklist
done

utils/subset_data_dir_tr_cv.sh --cv-spk-list data/giellagas-only/staff/spklist data/giellagas data/giellagas-only/interviewees data/giellagas-only/staff

utils/subset_data_dir.sh --per-spk data/giellagas-only/interviewees 140 data/giellagas-only/interviewees-train

utils/combine_data.sh data/giellagas-only/train data/giellagas-only/staff data/giellagas-only/interviewees-train

mkdir -p data/giellagas-only/interviewees-nontrain
utils/filter_scp.pl --exclude data/giellagas-only/interviewees-train/utt2spk \
  <(cut -f1 -d" " data/giellagas-only/interviewees/utt2spk) \
  >data/giellagas-only/interviewees-nontrain/uttlist

utils/subset_data_dir.sh --utt-list data/giellagas-only/interviewees-nontrain/uttlist \
  data/giellagas-only/interviewees data/giellagas-only/interviewees-nontrain

utils/subset_data_dir.sh --per-spk data/giellagas-only/interviewees-nontrain 10 data/giellagas-only/valid

mkdir -p data/giellagas-only/test
utils/filter_scp.pl --exclude data/giellagas-only/valid/utt2spk \
  <(cut -f1 -d" " data/giellagas-only/interviewees-nontrain/utt2spk) \
  >data/giellagas-only/test/uttlist

utils/subset_data_dir.sh --utt-list data/giellagas-only/test/uttlist \
  data/giellagas-only/interviewees-nontrain data/giellagas-only/test
