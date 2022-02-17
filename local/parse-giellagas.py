#!/usr/bin/env python3
import pympi
import pathlib
from collections import namedtuple
import warnings
import re

correction_marker = "¤"
spoken_noise = "<UNK>"
noise_matcher = re.compile(r"((\[[^\]]*\])|(\([^\)]*\)))")

Utterance = namedtuple("Utterance", ["start", "stop", "text", "spkid"])

# Linguistic type 'speech' is the Sami speech
# There are also English and Finnish translations
#NOTE: utterance times in milliseconds
def get_utterances(eafpath):
    data = pympi.Eaf(eafpath)
    for tiername in data.get_tier_ids_for_linguistic_type('speech'):
        params = data.get_parameters_for_tier(tiername)
        parts = data.get_annotation_data_for_tier(tiername)
        for part in parts:
            utterance = Utterance(
                    start=part[0], 
                    stop=part[1],
                    text=part[2],
                    spkid=params["PARTICIPANT"]
            )
            yield utterance

def process_text(text):
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace('"', "")
    text = text.strip()
    text = noise_matcher.sub("", text)
    processed = []
    for word in text.split():
        if word.endswith("-") or "ä" in word.lower() or "ö" in word.lower() or "å" in word.lower():
            processed.append(spoken_noise)
            continue
        if word.startswith(correction_marker):
            continue
        if len(word) > 1 and word.isupper():
            for char in word:
                processed.append(char.lower())
            continue
        if word.startswith("(") and word.endswith(")"):
            word = word[1:-1]
        word = word.lower()
        word = "".join(char for char in word if char.isalpha())
        processed.append(word)
    return " ".join(processed)

def extend_datadir(from_stream, recid, datadir):
    datadir = pathlib.Path(datadir)
    with open(datadir / "segments", "a") as segments, \
         open(datadir / "text", "a") as text, \
         open(datadir / "utt2spk", "a") as utt2spk:
        uttids = {}
        for utterance in from_stream:
            processed_text = process_text(utterance.text)
            if not processed_text:
                continue
            # Hack this here, IV_m4 speaks Finnish:
            if utterance.spkid == "IV_m4":
                continue
            uttid = f"{utterance.spkid}-{recid}-{utterance.start}-{utterance.stop}"
            if uttid in uttids:
                warnings.warn(f"Ignoring duplicate utterance (total overlap) {uttid}")
            print(uttid, utterance.spkid, file=utt2spk)
            print(uttid, processed_text, file=text)
            print(uttid, 
                    recid, 
                    "{:.2f}".format(utterance.start/1000), 
                    "{:.2f}".format(utterance.stop/1000), 
                    file=segments)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Giellagas-north Sami language files parser")
    parser.add_argument("input", help="The input .eaf file")
    parser.add_argument("recid", help="The recording ID to which the segment output is connected")
    parser.add_argument("outdir", help="""Directory where output goes. 
        Appends to outdir/text, outdir/utt2spk and outdir/segments""",
        type=pathlib.Path)
    args = parser.parse_args()
    utterance_stream = get_utterances(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    extend_datadir(utterance_stream, args.recid, args.outdir)

