#!/usr/bin/env/python3
"""Finnish Parliament ASR
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import webdataset as wds
from glob import glob
import io
import torchaudio
import tqdm
from pychain import ChainGraph, ChainGraphBatch 
import simplefst
import pathlib

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class LFMMIAM(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.wav
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs = self.modules.env_corrupt(wavs, wav_lens)

        feats = self.modules.wav2vec2(wavs)
        if self.hparams.subsampling == 2:
            pass
        elif self.hparams.subsampling == 3:
            feats = torch.repeat_interleave(feats,2,dim=1)[:,::self.hparams.subsampling,:]
        elif self.hparams.subsampling == 4:
            feats = feats[:,::2,:]
        encoded = self.modules.enc(feats)
        lfmmi_out = self.modules.lfmmi_lin_out(encoded)
        return lfmmi_out

    def compute_objectives(self, predictions, batch, stage):
        graphs = batch.graph
        wavs, wav_lens = batch.wav
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            pass
            #graphs = graphs + graphs
            #alis = torch.cat([alis, alis], dim=0)
            #ali_lens = torch.cat([ali_lens, ali_lens])
            #wav_lens = torch.cat([wav_lens, wav_lens])
        lfmmi_out = predictions
        num_transitions = list(map(self.hparams.transgetter, batch.graph))
        output_lengths = (lfmmi_out.shape[1] * wav_lens).int().cpu()
        max_num_states = max(map(self.hparams.stategetter, batch.graph))
        numerator_graphs = ChainGraphBatch(
                graphs,
                max_num_transitions=max(num_transitions),
                max_num_states=max_num_states
        )
        lfmmi_loss = self.hparams.chain_loss(lfmmi_out, output_lengths, numerator_graphs)
        return lfmmi_loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            pass

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        # Summarize the statistics from the stage for record-keeping.
        else:
            pass

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"],}, 
                min_keys=["loss"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "wav2vec_opt", self.wav2vec_optimizer
                )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    
    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        should_step = self.step % self.hparams.grad_accumulation_factor == 0
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        (loss / self.hparams.grad_accumulation_factor).backward()

        if should_step:
            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()
        
            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        return loss.detach()


    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if getattr(self.hparams, "avg_ckpts", 1) > 1:
            ckpts = self.checkpointer.find_checkpoints(
                    max_key=max_key,
                    min_key=min_key,
                    max_num_checkpoints=self.hparams.avg_ckpts
            )
            model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    ckpts, "model" 
            )
            self.hparams.model.load_state_dict(model_state_dict)
            self.checkpointer.save_checkpoint(name=f"AVERAGED-{self.hparams.avg_ckpts}")

def numfsts_to_local_tmp(fstdir, tmpdir):
    """Copies the chain numerator FSTs onto a local disk"""
    fstdir = pathlib.Path(fstdir)
    tmpdir = pathlib.Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    sb.utils.superpowers.run_shell(f"rsync --update {fstdir}/num.*.ark {tmpdir}/")
    numfsts = {}
    for scpfile in fstdir.glob("num.*.scp"):
        with open(scpfile) as fin:
            for line in fin:
                uttid, data = line.strip().split()
                arkpath, offset = data.split(":")
                newpath = arkpath.replace(str(fstdir), str(tmpdir))
                numfsts[uttid] = (newpath, int(offset))
    return numfsts

def dataio_prepare(hparams, numfsts):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys mapping to 
        WebDataset datasets dataloaders for them.
    """
    TRAIN_FSTS = {}
    # READ train fsts to mem:
    if getattr(hparams, "train_fsts_in_mem", False):
        for uttid, (fstpath, offset) in numfsts["train"].items():
            TRAIN_FSTS[uttid] = ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)
    def load_train_fst(sample):
        uttid = sample["__key__"]
        if uttid in TRAIN_FSTS:
            sample["graph"] = TRAIN_FSTS[uttid] 
        else:
            fstpath, offset = numfsts["train"][uttid]
            sample["graph"] = ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)
        return sample

    def load_valid_fst(sample):
        uttid = sample["__key__"]
        fstpath, offset = numfsts["valid"][uttid]
        sample["graph"] = ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)
        return sample

    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(wav="audio.pth")
            .map(load_train_fst, handler=wds.warn_and_continue)
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(wav="audio.pth")
            .map(load_valid_fst, handler=wds.warn_and_continue)
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                drop_end=False,
                **hparams["valid_dynamic_batch_kwargs"],
            )
    )
    return {"train": traindata, "valid": validdata}




if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Copy numerator FSTs to local drive:
    numfsts = {}
    numfsts["train"] = numfsts_to_local_tmp(hparams["numfstdir"], hparams["numfsttmpdir"])
    numfsts["valid"] = numfsts_to_local_tmp(hparams["valid_numfstdir"], hparams["valid_numfsttmpdir"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, numfsts)
    # read valid data into memory:
    datasets["valid"] = torch.utils.data.DataLoader(
            list(iter(datasets["valid"])),
            batch_size=None
    )

    # Pretrain if defined:
    if "pretrainer" in hparams:
        if "pretrain_max_key" in hparams:
            ckpt = hparams["ckpt_finder"].find_checkpoint(max_key=hparams["pretrain_max_key"])
        elif "pretrain_min_key" in hparams:
            ckpt = hparams["ckpt_finder"].find_checkpoint(min_key=hparams["pretrain_min_key"])
        else:
            ckpt = hparams["ckpt_finder"].find_checkpoint()
        hparams["pretrainer"].collect_files(ckpt.path)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = LFMMIAM(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
       train_loader_kwargs = hparams["train_loader_kwargs"]
    )
