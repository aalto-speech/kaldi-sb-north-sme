#!/usr/bin/env python3
"""Write Kaldi data as WebDataset shards"""

import webdataset as wds
import multiprocessing as mp
import subprocess
import torchaudio
import queue
import os
import pathlib
import warnings
import shutil
import kaldi_io
import torch
import numpy as np
import locale
import more_itertools


def kaldi_map_stream(path):
    with open(path) as fi:
        for line in fi:
            try:
                uttid, data = line.strip().split(maxsplit=1)
            except ValueError:
                # Empty entry
                uttid = line.strip().split(maxsplit=1)
                data = ""
            yield uttid, data

def read_rxwav(data):
    if data.endswith("|"):
        with subprocess.Popen(data[:-1], shell=True, stdout=subprocess.PIPE) as proc:
            signal, samplerate = torchaudio.load(proc.stdout)
    elif ".ark:" in data:
        f = kaldi_io.open_or_fd(data)
        signal, samplerate = torchaudio.load(f)
        f.close()
    else:
        signal, samplerate = torchaudio.load(data)
    return signal.squeeze(0), samplerate

def segments_to_output(segments_path, wavscp_path, fade_len=0.005):
    wavscp = dict(kaldi_map_stream(wavscp_path))
    current_id = None
    current_wav = None,
    current_samplerate = None
    current_fader = None
    for uttid, segment_data in kaldi_map_stream(segments_path):
        wav_id, start, end = segment_data.split()
        if wav_id != current_id:
            current_wav, current_samplerate = read_rxwav(
                    wavscp[wav_id]
            )
            current_id = wav_id
            fade_time = int(fade_len*current_samplerate)
            current_fader = torchaudio.transforms.Fade(
                    fade_in_len=fade_time, 
                    fade_out_len=fade_time, 
            )
        start_ind = int(float(start) * current_samplerate)
        end_ind = int(float(end) * current_samplerate)
        output = {"audio.pth": current_fader(current_wav[start_ind:end_ind]),
                "meta.json": {"samplerate": current_samplerate}}
        yield uttid, output 

def wavscp_to_output(wavscp_path):
    for uttid, wavdata in kaldi_map_stream(wavscp_path):
        signal, samplerate = read_rxwav(wavdata)
        output = {"audio.pth": signal,
                "meta.json": {"samplerate": samplerate}}
        yield uttid, output

def text_to_output(text_path):
    for uttid, data in kaldi_map_stream(text_path):
        output = {"transcript.txt": data}
        yield uttid, output

def utt2spk_to_output(utt2spk_path):
    for uttid, data in kaldi_map_stream(utt2spk_path):
        output = {"meta.json": {"spkid":data}}
        yield uttid, output

def featsscp_to_output(featsscp_path):
    for uttid, feats in kaldi_io.read_mat_scp(featsscp_path):
        output = {"feats.pth": torch.from_numpy(feats)}
        yield uttid, output

def ali_to_output(aliark_path):
    for uttid, ali in kaldi_io.read_ali_ark(aliark_path):
        # This conversion is needed, see:
        # https://github.com/pytorch/audio/blob/8a347b62cf5c907d2676bdc983354834e500a282/torchaudio/kaldi_io.py#L59-L61
        ali = np.ascontiguousarray(ali)
        output = {"ali.pth": torch.from_numpy(ali)}
        yield uttid, output


def sync_streams(streams, maxskip=100):
    streams = [more_itertools.peekable(stream) for stream in streams]
    skipped = 0
    duplicated = 0
    buffers = [None for stream in streams]
    while True:
        for i, buf in enumerate(buffers):
            if buf is None:
                try:
                    buffers[i] = next(streams[i])
                except StopIteration:
                    return
        # NOTE:
        # Kaldi convention is to use the C locale.
        # It could happen that you require a different
        # locale for running this script (as you might have
        # text in UTF-8 encoding for example).
        # By design this does naive comparison (not locale-aware),
        # which should match the C locale, as far as I know.
        key_to_keep = max(key for key, value in buffers)
        for i, (key, value) in enumerate(buffers):
            if key != key_to_keep:
                buffers[i] = None
        if all(buf is not None for buf in buffers):
            # Allowing duplicates requires renaming:
            if duplicated > 0:
                yield [(key + f"-{duplicated}", value) 
                        for (key, value) in buffers]
            else:
                yield buffers
            skipped = 0
            found_duplicate = False
            for i, stream in enumerate(streams):
                if stream.peek((None, None))[0] == key_to_keep:
                    buffers[i] = None
                    found_duplicate = True
            if not found_duplicate:
                buffers = [None for stream in streams]
                duplicated = 0
            else:
                duplicated += 1
        else:
            skipped += 1
            if skipped > maxskip:
                MSG = "Skipped too many partially available utterances!"
                raise RuntimeError(MSG)


def feat_ali_chunks_to_output(featsscp_path, aliark_path, chunklen, subsampling, contextlen):
    chunklen = int(chunklen)
    subsampling = int(subsampling)
    contextlen = int(contextlen)
    ali_chunklen = chunklen // subsampling
    feat_stream = featsscp_to_output(featsscp_path)
    ali_stream = ali_to_output(aliark_path)
    for (key, feats_output), (_, ali_output) in sync_streams([feat_stream, ali_stream]):
        feats = feats_output['feats.pth']
        ali = ali_output['ali.pth']
        padded_feats = torch.cat(
                (
                    feats[0].unsqueeze(0).repeat_interleave(contextlen,dim=0),
                    feats,
                    feats[-1].unsqueeze(0).repeat_interleave(contextlen,dim=0)
                )
        )
        for i in range(ali.shape[0] // ali_chunklen):
            feats_chunk = padded_feats[i*chunklen:(i+1)*chunklen+2*contextlen,:]
            if feats_chunk.shape[0] != (chunklen + 2*contextlen):
                # Skip chunks with bad boundary conditions :(
                continue
            ali_chunk = ali[i*ali_chunklen:(i+1)*ali_chunklen]
            output = {'feats.pth': feats_chunk,
                    'ali.pth': ali_chunk}
            yield key, output


STREAM_FUNCS = {
        "text": text_to_output,
        "segments": segments_to_output,
        "wavscp": wavscp_to_output,
        "utt2spk": utt2spk_to_output,
        "featsscp": featsscp_to_output,
        "aliark": ali_to_output,
        "featalichunk": feat_ali_chunks_to_output,
}


def make_data_point(outputs):
    data_point = {}
    for uttid, output in outputs:
        # HACK: WebDataset cannot handle periods in uttids:
        uttid = uttid.replace(".", "")
        if "__key__" not in data_point:
            data_point["__key__"] = uttid
        elif uttid != data_point["__key__"]:
            MSG = "Mismatched key, data probably not "
            MSG += "sorted and filtered the same way! "
            MSG += f"Conflict: {uttid} != {data_point['__key__']}; "
            MSG += f"{' '.join(output.keys())} did not "
            MSG += f"match with {' '.join(data_point.keys())}"
            raise RuntimeError(MSG)
        for key, data in output.items():
            if isinstance(data, dict):
                to_update = data_point.setdefault(key, {})
                to_update.update(data)
            else:
                data_point[key] = data
    return data_point


def make_streams(sources):
    streams = []
    for name, args in sources.items():
        stream = STREAM_FUNCS[name](*args)
        streams.append(stream)
    return streams
    
            

SHARD_DEFAULTS = {
        "maxcount":100000,
        "maxsize": 3e9,
        "compress": True,
}



def write_shards(shard_dir, source_queue, shard_kwargs=SHARD_DEFAULTS):
    shard_dir = pathlib.Path(shard_dir)
    shard_dir.mkdir(parents=True)
    shardpattern = f"{shard_dir}/shard-%06d.tar"
    try:
        sources = source_queue.get(timeout=5)
    except queue.Empty:
        return
    with wds.ShardWriter(shardpattern, **shard_kwargs) as fo:
        while True:
            streams = make_streams(sources)
            for outputs in sync_streams(streams):
                data_point = make_data_point(outputs)
                fo.write(data_point)
            try:
                sources = source_queue.get(timeout=5)
            except queue.Empty:
                break

def fill_queue(nj, sources):
    source_queue = mp.Queue()
    for jobid in range(1, nj+1):
        jobsources = {}
        for name, args in sources.items():
            jobsources[name] = tuple(arg.replace("JOB", str(jobid))
                    for arg in args)
        source_queue.put(jobsources)
    return source_queue

def process_queue_in_parallel(num_proc, shard_dir, source_queue, shard_kwargs=SHARD_DEFAULTS):
    shard_dir = pathlib.Path(shard_dir)
    processes = []
    if num_proc > 0:
        for i in range(num_proc):
            proc = mp.Process(
                    target=write_shards,
                    kwargs={
                        "shard_dir": shard_dir / str(i),
                        "source_queue": source_queue,
                        "shard_kwargs": shard_kwargs
                    }
            )
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()
    else:
        # No parallel processing, use this process!
        write_shards(
                shard_dir = shard_dir / "0",
                source_queue = source_queue,
                shard_kwargs = shard_kwargs
        )



def collect_shards(split_shard_dir, target_dir):
    shard_split_dir = pathlib.Path(split_shard_dir)
    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    for shard_dir in shard_split_dir.iterdir():
        for shard in shard_dir.iterdir():
            shutil.move(shard, target_dir / ("shard-%06d.tar" % i))
            i += 1
        shard_dir.rmdir()
    shard_split_dir.rmdir()


if __name__ == "__main__":
    import argparse
    import inspect
    parser = argparse.ArgumentParser()
    parser.add_argument("nj",
            help="""The number of Kaldi archives,
            this is called nj (num jobs) after the 
            Kaldi convention""",
            type=int)
    parser.add_argument("shard_dir",
            help="""The top-level directory where 
            the shards should go.""",
            type=pathlib.Path)
    parser.add_argument("--num-proc",
            help="""The number of processes to use""",
            default=4,
            type=int)
    for name, func in STREAM_FUNCS.items():
        spec = inspect.getfullargspec(func)
        if spec.defaults is None:
            nargs = len(spec.args)
        else:
            nargs = len(spec.args) - len(spec.defaults)
        parser.add_argument(f"--{name}",
            nargs = nargs,
            metavar = tuple(spec.args[:nargs])
        )
    for name, default in SHARD_DEFAULTS.items():
        parser.add_argument(f"--shard-{name}",
                default=default,
                type=type(default)
        )
    args = parser.parse_args()
    sources = {name: getattr(args,name) for name in STREAM_FUNCS
                if getattr(args,name) is not None}
    shard_kwargs = {name: getattr(args, f"shard_{name}") for name in SHARD_DEFAULTS}
    source_queue = fill_queue(args.nj, sources)
    process_queue_in_parallel(args.num_proc, args.shard_dir / "TMP", source_queue, shard_kwargs)
    collect_shards(args.shard_dir / "TMP", args.shard_dir) 
