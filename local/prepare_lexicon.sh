#!/bin/bash
# Prepares a grapheme lexicon based on the training data text
#
# This script can also add lexicon entries for extra text files' words
# The script checks that the extra text entries don't have graphemes
# that don't appear in the training data

oov_entry="<UNK>"
extra_texts=

. path.sh
. parse_options.sh

echo "$0 $@"  # Print the command line for logging

if [ $# -ne 3 ]; then
	echo "Usage: local/prepare_lexicon.sh <traindir> <workdir> <outdir>"
	echo "e.g.: $0 data/train data/local/dict_train data/lang_train"
	echo
	echo "Prepare a grapheme unit lexicon. Look at top of the script"
	echo "for options."
	echo
	exit 1
fi

traindir=$1
workdir=$2
outdir=$3

mkdir -p $workdir

local/check_grapheme_sets_compatible.py $traindir/text $extra_texts || exit 1

local/seed_dict.sh $workdir

cat $traindir/text $extra_texts | cut -d" " -f2- | tr " " "\n" | sort -u | local/word-list-to-lexicon.py - >> $workdir/lexicon.txt
cut -d" " -f2- $workdir/lexicon.txt | tr " " "\n" | sort -u | grep -vf $workdir/silence_phones.txt - > $workdir/nonsilence_phones.txt

tmpdir=$(mktemp -d)

utils/prepare_lang.sh $workdir "$oov_entry" $tmpdir $outdir

rm -r $tmpdir

echo "Finished lexicon preparation!"

