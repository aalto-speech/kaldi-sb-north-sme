#!/usr/bin/env python3
from collections import namedtuple
import pathlib

Segment = namedtuple("Segment", ["uttid", "recid", "start", "stop", "text"])

def produce_segments(ctmfile, boundary_marker, margin=0.3, rec2dur={}):
    min_segment_len = margin*3  # An arbitrary but probably decent filter
    with open(ctmfile) as fi:
        ongoing_start = None
        ongoing_stop = None
        ongoing_recid = None
        ongoing_text = []
        onrec_index = 1
        for line in fi:
            recid, channel, start, duration, token = line.strip().split()
            if ongoing_recid != recid:
                if ongoing_recid is not None and ongoing_text:
                    uttid = ongoing_recid + "-{:03d}".format(onrec_index)
                    if ongoing_recid in rec2dur:
                        ongoing_stop = min(float(ongoing_stop)+margin, rec2dur[ongoing_recid])
                    # filter very short utterances:
                    if ongoing_stop - ongoing_start > min_segment_len:
                        yield Segment(uttid=uttid,
                                recid=ongoing_recid,
                                start=ongoing_start,
                                stop=ongoing_stop,
                                text=ongoing_text)
                ongoing_start = max(float(start)-margin, 0.)
                ongoing_stop = float(start) + float(duration)
                ongoing_recid = recid
                ongoing_text = [token] if token != boundary_marker else []
                onrec_index = 1
            elif token == boundary_marker:
                if ongoing_text:
                    uttid = ongoing_recid + "-{:03d}".format(onrec_index)
                    onrec_index += 1
                    ongoing_stop = min(float(ongoing_stop)+margin,
                            max(float(start)-margin/2, float(ongoing_stop)))
                    if ongoing_stop - ongoing_start > min_segment_len:
                        yield Segment(uttid=uttid,
                                recid=ongoing_recid,
                                start=ongoing_start,
                                stop=ongoing_stop,
                                text=ongoing_text)
                ongoing_start = min(max(float(start)-margin, ongoing_stop+margin/2), float(start))
                ongoing_stop = float(start) + float(duration)
                ongoing_text = []
            else:
                ongoing_text.append(token)
                ongoing_stop = float(start) + float(duration)
        if ongoing_text:
            uttid = ongoing_recid + "-{:03d}".format(onrec_index)
            if ongoing_recid in rec2dur:
                ongoing_stop = min(float(ongoing_stop)+margin, rec2dur[ongoing_recid])
            yield Segment(uttid=uttid,
                    recid=ongoing_recid,
                    start=ongoing_start,
                    stop=ongoing_stop,
                    text=ongoing_text)


def overwrite_datadir(from_stream, rec2spk, datadir):
    datadir = pathlib.Path(datadir)
    with open(datadir / "segments", "w") as segments, \
         open(datadir / "text", "w") as text, \
         open(datadir / "utt2spk", "w") as utt2spk:
        for segment in from_stream:

            spkid = rec2spk[segment.recid]
            print(segment.uttid, spkid, file=utt2spk)
            print(segment.uttid, " ".join(segment.text), file=text)
            print(segment.uttid, 
                    segment.recid, 
                    "{:.2f}".format(segment.start), 
                    "{:.2f}".format(segment.stop), 
                    file=segments)

def read_rec2dur(path):
    rec2dur = {}
    with open(path) as fi:
        for line in fi:
            recid, dur = line.strip().split()
            rec2dur[recid] = float(dur)
    return rec2dur

def read_rec2spk(path):
    rec2spk = {}
    with open(path) as fi:
        for line in fi:
            recid, spk = line.strip().split()
            rec2spk[recid] = spk 
    return rec2spk

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ctmfile", help = "The file to operate on")
    parser.add_argument("rec2spk", help = "The old utt2spk, the file that maps new recid to spkid")
    parser.add_argument("outdir", help = "Where to write the output")
    parser.add_argument("--rec2dur", help = "The old utt2dur, the file that maps new recid to recording duration. Optional, used to determine how far into the file it is safe to read (last utterance silence margin).")
    parser.add_argument("--utterance-boundary", 
            default = "<SENT>",
            help = "Boundary marker, at which to split segments"
    )
    parser.add_argument("--margin", 
            help = "Silence margin (in seconds) to try and leave in the segments", 
            type=float, 
            default=0.3
    )
    args = parser.parse_args()
    if args.rec2dur is not None:
        rec2dur = read_rec2dur(args.rec2dur)
    else:
        rec2dur = {}
    rec2spk = read_rec2spk(args.rec2spk)
    segment_stream = produce_segments(args.ctmfile, args.utterance_boundary, args.margin, rec2dur)
    overwrite_datadir(segment_stream, rec2spk, args.outdir)

