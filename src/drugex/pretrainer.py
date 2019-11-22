#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for pre-training and fine-tuning the RNN network.

In this project, it is trained on ZINC set and A2AR set collected by
dataset.py. In the end, RNN model can generate molecule library.
"""
import torch

import os

import click

from drugex import util
from drugex.api.corpus import CorpusCSV
from drugex.api.pretrain.generators import BasicGenerator


def _main_helper(*, input_directory, batch_size, epochs_pr, epochs_ex, output_directory, use_tqdm=False):
    # Construction of the pretrainer corpus
    voc_file = os.path.join(input_directory, "voc.txt")
    zinc_corpus = os.path.join(input_directory, "zinc_corpus.txt")
    pre_corpus = CorpusCSV.fromFiles(corpus_path=zinc_corpus, vocab_path=voc_file)

    # Pre-training the RNN model with ZINC set
    prior = BasicGenerator(
        pre_corpus
        , out_dir=output_directory
        , out_identifier="pr"
        , train_params={
            "epochs" : epochs_pr
        }
    )
    if not os.path.exists(prior.net_pickle_path):
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
    ex_corpus = CorpusCSV.fromFiles(corpus_path=chembl_corpus, vocab_path=voc_file)
    ser = BasicGenerator.BasicDeserializer(
        ex_corpus
        , out_dir=output_directory
        , in_dir=output_directory
        , in_identifier="pr"
        , out_identifier="ex"
        , train_params={
            "epochs" : epochs_ex
        }
    )
    explore = BasicGenerator.load(ser)

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
    if gpu:
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
