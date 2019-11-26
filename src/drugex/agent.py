#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file is used for generator training under reinforcement learning framework.

It is implemented by integrating exploration strategy into REINFORCE algorithm.
The deep learning code is implemented by PyTorch ( >= version 1.0)
"""

import os

import click
import torch
from rdkit import rdBase

from drugex.api.agent.agents import DrugExAgentTrainer
from drugex.api.agent.callbacks import BasicAgentMonitor
from drugex.api.agent.policy import PG
from drugex.api.environ.serialization import FileEnvDeserializer
from drugex.api.model.callbacks import BasicMonitor
from drugex.api.pretrain.generators import BasicGenerator

def _main_helper(*, epsilon, baseline, batch_size, mc, vocabulary_path, output_dir):
    # Environment (predictor)
    des = FileEnvDeserializer(output_dir, 'RF_cls_ecfp6')
    environ = des.getModel()

    # Agent (generator, exploitation network)
    exploit_monitor = BasicMonitor(output_dir, "pr")
    exploit = BasicGenerator(initial_state=exploit_monitor)

    # exploration network
    explore_monitor = BasicMonitor(output_dir, "ex")
    explore = BasicGenerator(initial_state=explore_monitor)

    policy = PG(batch_size, mc, epsilon, beta=baseline)
    identifier = 'e_%.2f_%.1f_%dx%d' % (policy.epsilon, policy.beta, policy.batch_size, policy.mc)
    agent_monitor = BasicAgentMonitor(output_dir, identifier)
    agent = DrugExAgentTrainer(
        agent_monitor
        , environ
        , exploit
        , policy
        , explore
        , {"n_epochs" : 1000}
    )
    agent.train()


@click.command()
@click.option('-d', '--input-directory', type=click.Path(file_okay=False, dir_okay=True), show_default=True, default="data")
@click.option('-o', '--output-directory', type=click.Path(file_okay=False, dir_okay=True), show_default=True, default="output")
@click.option('--mc', type=int, default=10, show_default=True)
@click.option('-s', '--batch-size', type=int, default=512, show_default=True)
@click.option('-t', '--num-threads', type=int, default=1, show_default=True)
@click.option('-e', '--epsilon', type=float, default=0.1, show_default=True)
@click.option('-b', '--baseline', type=float, default=0.1, show_default=True)
@click.option('-g', '--gpu', type=int, default=0)
def main(input_directory, output_directory, mc, batch_size, num_threads, epsilon, baseline, gpu):
    rdBase.DisableLog('rdApp.error')
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available() and gpu:
        torch.cuda.set_device(gpu)
    _main_helper(
        baseline=baseline,
        batch_size=batch_size,
        mc=mc,
        epsilon=epsilon,
        vocabulary_path=os.path.join(input_directory, "voc.txt"),
        output_dir=output_directory,
    )


if __name__ == "__main__":
    main()
