#!/bin/bash
set -eu

# NOTE: first run local/giellagas-only/make-split.sh

mkdir -p data/non-indep

utils/combine_data.sh data/non-indep/train data/uit-sme-segmented/ data/giellagas-only/train

ln -s "$PWD"/data/giellagas-only/valid "$PWD"/data/non-indep/valid
ln -s "$PWD"/data/giellagas-only/test "$PWD"/data/non-indep/test

