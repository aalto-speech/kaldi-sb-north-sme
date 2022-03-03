#!/bin/bash
# This script creates a dict directory for utils/prepare_lang.sh
# It also creates the important lexicon_placeholders.txt file
# which is used by subword-kaldi/local/make_lfst_spm.py

set -eu

unk="<UNK>"
phone_table="data/lang_train/phones.txt"
prepare_lang_opts=

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
   echo "Usage: $0 vocab dict_out lang_out"
   echo "e.g.:  $0 data/train/vocab data/dict data/lang_spm"
   exit 1;
fi

vocab=$1
dictdir=$2
langdir=$3

mkdir -p $dictdir 
mkdir -p tmp
tmpdir=$(mktemp -d -p tmp)

local/seed_dict.sh "$dictdir"

# Filter special tokens from sentencepiece vocab 
# NOTE: sentencepiece uses lowercase <unk>
sed -e '/^<unk>$/d' -e '/^<s>$/d' -e '/^<\/s>$/d' -e '/^▁$/d' <${vocab} > ${tmpdir}/vocab.filtered

subword-kaldi/local/make_spm_lexicon.py \
  --g2p-cmd "local/word-list-to-lexicon.py {filepath}" \
   ${tmpdir}/vocab.filtered > ${dictdir}/lexicon.txt 

# Add <unk> and the sentencepiece space back. SOS/EOS are added by utils/prepare_lang.sh
echo -e "$unk\tSPN" >> ${dictdir}/lexicon.txt
echo -e "▁\tSIL" >> ${dictdir}/lexicon.txt


# Used to do it like this:
# Filter the lexicon for a list of phones.
#cut -f2- < ${dictdir}/lexicon.txt | tr ' ' '\n' | sed 's/^[ \t]*//;s/[ \t]*$//' | sed '/^$/d' | sort -u | grep -v -F -f ${dictdir}/silence_phones.txt > ${dictdir}/nonsilence_phones.txt

# Now use the given phone_table instead:
cut -f1 -d" " <"$phone_table" | sed 's/_.$//g' | sed '/^\#/d' | sed '/^<eps>/d' | sort -u | grep -v -F -f ${dictdir}/silence_phones.txt > ${dictdir}/nonsilence_phones.txt


subword-kaldi/local/make_spm_lexicon.py \
  --g2p-cmd "local/word-list-to-lexicon.py {filepath}" \
  --add-placeholders \
  ${tmpdir}/vocab.filtered > ${dictdir}/lexicon_placeholders.txt

# Add just <unk>  to placeholder lexicon. The sentencepiece space is handled by subword-kaldi/local/make_lfst_spm.py
echo -e "$unk\tSPN" >> ${dictdir}/lexicon_placeholders.txt

# Now, dict directory is ready.
rm -Rf ${tmpdir}

extra=5 #Need 5 extra disambig symbols for sentencepiece
utils/prepare_lang.sh --phone_symbol_table "$phone_table" $prepare_lang_opts --num-extra-phone-disambig-syms $extra ${dictdir} "$unk" ${langdir}/local ${langdir} 

# Overwrite L_disambig.fst
subword-kaldi/local/make_lfst_spm.py $(tail -n$extra ${langdir}/phones/disambig.txt) \
  --lexicon-file ${langdir}/local/lexiconp_disambig.txt ${dictdir}/lexicon_placeholders.txt |\
  fstcompile --isymbols=${langdir}/phones.txt --osymbols=${langdir}/words.txt --keep_isymbols=false --keep_osymbols=false |\
  fstaddselfloops  ${langdir}/phones/wdisambig_phones.int ${langdir}/phones/wdisambig_words.int |\
  fstarcsort --sort_type=olabel > ${langdir}/L_disambig.fst

echo "Done making sentencepiece Lang dir" 
