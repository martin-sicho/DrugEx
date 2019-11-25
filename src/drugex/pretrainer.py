#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for pre-training and fine-tuning the RNN network.

In this project, it is trained on ZINC set and A2AR set collected by
dataset.py. In the end, RNN model can generate molecule library.
"""
import torch

import os

import click

from drugex.core import util
from drugex.api.corpus import CorpusCSV
from drugex.api.model.callbacks import BasicMonitor
from drugex.api.pretrain.generators import BasicGenerator


def _main_helper(*, input_directory, batch_size, epochs_pr, epochs_ex, output_directory, use_tqdm=False):
    # Construction of the pretrainer corpus
    zinc_corpus = os.path.join(input_directory, "zinc_corpus.txt")
    pre_corpus = CorpusCSV.fromFiles(corpus_path=zinc_corpus)

    # Pre-training the RNN model with ZINC set
    pr_logger = BasicMonitor(out_dir=output_directory, identifier="pr")
    prior = BasicGenerator(
        monitor=pr_logger
        , corpus=pre_corpus
        , train_params={
            "epochs" : epochs_pr
        }
    )
    if not pr_logger.best_state:
        print('Exploitation network begins to be trained...')
        prior.pretrain(train_loader_params={
            "batch_size" : batch_size
            , "shuffle" : True
            , "drop_last" : True
            , "collate_fn" : util.MolData.collate_fn
        })
        print('Exploitation network training is finished!')

    # Fine-tuning the RNN model with A2AR set as exploration stragety
    chembl_corpus = os.path.join(input_directory, 'chembl_corpus.txt')
    ex_corpus = CorpusCSV.fromFiles(corpus_path=chembl_corpus)
    ex_logger = BasicMonitor(out_dir=output_directory, identifier="ex")
    explore = BasicGenerator(
        monitor=ex_logger
        , corpus=ex_corpus
        , initial_state=pr_logger
        , train_params={
            "epochs" : epochs_ex
        }
    )
    print('Exploration network begins to be trained...')
    explore.pretrain(
        train_loader_params={
            "batch_size" : batch_size
            , "collate_fn" : util.MolData.collate_fn
        }
        , validation_size=batch_size
        , valid_loader_params={
            "batch_size" : batch_size
            , "collate_fn" : util.MolData.collate_fn
        }
    )
    print('Exploration network training is finished!')


@click.command()
@click.option('-d', '--input-directory', type=click.Path(dir_okay=True, file_okay=False), default='data')
@click.option('-o', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), default='output')
@click.option('-b', '--batch-size', type=int, default=512, show_default=True)
@click.option('-p', '--epochs-pr', type=int, default=300, show_default=True)
@click.option('-e', '--epochs-ex', type=int, default=400, show_default=True)
@click.option('-g', '--gpu', type=int, default=0)
@click.option('--use-tqdm', is_flag=True)
def main(input_directory, output_directory, batch_size, epochs_pr, epochs_ex, gpu, use_tqdm):
    if gpu and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    _main_helper(
        input_directory=input_directory,
        output_directory=output_directory,
        batch_size=batch_size,
        epochs_pr=epochs_pr,
        epochs_ex=epochs_ex,
        use_tqdm=True if use_tqdm else False,
    )


if __name__ == "__main__":
    main()
