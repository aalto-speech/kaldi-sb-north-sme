#!/bin/bash
set -eu
. cmd.sh

stage=1
BPE_units=1000
varikn_scale=0.01
varikn_cmd="$train_cmd"
varikn_extra="--clear_history --3nzer --arpa"
skip_lang=false
lmdatadir="data/lm"

echo $0 $@

. path.sh
. parse_options.sh


if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <outdir>"
	echo "e.g.: $0 --BPE-units 1000 data/lang_bpe.1000.varikn"
  exit 1
fi

outdir="$1"

lmdir="exp/lm/varikn.bpe${BPE_units}.d${varikn_scale}"

if [ "$stage" -le 0 ]; then
  local/lm/prep-lm-data.sh
fi

if [ "$stage" -le 1 ]; then
  echo "Training SentencePiece BPE: $BPE_units"
  mkdir -p "$lmdir"
  $train_cmd "$lmdir"/log/spm_train_"$BPE_units".log \
    spm_train --input="$lmdatadir"/train.plain \
    --model_prefix="$lmdir"/bpe.$BPE_units \
    --vocab_size="$BPE_units" \
    --character_coverage=1.0 \
    --add_dummy_prefix=false \
    --model_type="bpe"

  # Vocab to plain vocab ( normal SPM format is <subword> <id> ) 
  cut -f1 "$lmdir"/bpe.$BPE_units.vocab > "$lmdir"/bpe.$BPE_units.vocab.plain

  $train_cmd "$lmdatadir"/log/spm_encode_"$BPE_units".log \
    spm_encode --model="$lmdir"/bpe."$BPE_units".model \
    --output_format=piece \< "$lmdatadir"/train.boundaries \> "$lmdatadir"/train.bpe.$BPE_units
  $train_cmd "$lmdatadir"/log/spm_encode_"$BPE_units"_valid.log \
    spm_encode --model="$lmdir"/bpe."$BPE_units".model \
    --output_format=piece \< "$lmdatadir"/valid.boundaries \> "$lmdatadir"/valid.bpe.$BPE_units
fi

if [ "$stage" -le 2 ]; then
  local/lm/train_varikn.sh \
		--cmd "$train_cmd"  \
		--scale "$varikn_scale" \
		--extra-args "$varikn_extra" \
    "cat $lmdatadir/train.bpe.$BPE_units" \
    "cat $lmdatadir/valid.bpe.$BPE_units" \
    "$lmdir" 
fi

if [ "$stage" -le 3 ]; then
  echo "Compute perplexity"
  perplexity --arpa "$lmdir"/varikn.lm.gz \
    "$lmdatadir"/valid.bpe."$BPE_units" \
    "$lmdir"/valid_perplexity
fi

if [ "$skip_lang" = true ]; then
	echo "Skipping lang dir creation."
	exit 0
fi

dict_dir="data/local/dict_lm_bpe.$BPE_units"
if [ "$stage" -le 4 ]; then
	echo "Make SentencePiece LM."
	local/lm/make_spm_lang.sh "$lmdir"/bpe.${BPE_units}.vocab.plain $dict_dir $outdir
fi

if [ "$stage" -le 5 ]; then
	echo "Convert ARPA to FST."
	utils/format_lm.sh \
		$outdir "$lmdir"/varikn.lm.gz \
		$dict_dir/lexicon.txt $outdir
fi

