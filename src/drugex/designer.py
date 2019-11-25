#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for *de novo* designing the molecule with well-trained RNN model."""
import os

import torch
import click

from drugex.api.agent.callbacks import BasicAgentMonitor
from drugex.api.designer.designers import CSVConsumer, BasicDesigner
from drugex.api.environ.serialization import FileEnvDeserializer
from drugex.api.pretrain.generators import BasicGenerator


def generate(agent_name, out_dir, environment_path, num, batch_size):
    agent = BasicGenerator(initial_state=BasicAgentMonitor(out_dir, agent_name))
    des = FileEnvDeserializer(environment_path)
    environ = des.getModel()
    consumer = CSVConsumer(os.path.join(out_dir, 'designer_mols.csv'))

    designer = BasicDesigner(agent, consumer, environ, n_samples=num, batch_size=batch_size)
    designer()
    consumer.save()

@click.command()
@click.option('-a', '--agent-name', required=True)
@click.option('-o', '--out-dir', required=True)
@click.option('-e', '--environment-path', default="RF_cls_ecfp6.pkg")
@click.option('-n', '--number-smiles', type=int, default=10000, show_default=True)
@click.option('-b' , '--batch-size', type=int, default=500, show_default=True)
@click.option('-g', '--gpu', type=int, default=0)
def main(agent_name, out_dir, environment_path, number_smiles, batch_size, gpu):

    if torch.cuda.is_available() and gpu:
        torch.cuda.set_device(gpu)

    generate(
        agent_name=agent_name,
        environment_path=environment_path,
        out_dir=out_dir,
        batch_size=batch_size,
        num=number_smiles,
    )

    # main('v1/net_e_5_1_500x10.pkg', 'v1/mol_e_5_1_500x10.txt')
    # main('v1/net_e_10_1_500x10.pkg', 'v1/mol_e_10_1_500x10.txt')
    # main('v1/net_e_15_1_500x10.pkg', 'v1/mol_e_15_1_500x10.txt')
    # main('v1/net_e_20_1_500x10.pkg', 'v1/mol_e_20_1_500x10.txt')
    # main('v1/net_e_25_1_500x10.pkg', 'v1/mol_e_25_1_500x10.txt')
    #
    # main('v1/net_e_5_0_500x10.pkg', 'v1/mol_e_5_0_500x10.txt')
    # main('v1/net_e_10_0_500x10.pkg', 'v1/mol_e_10_0_500x10.txt')
    # main('v1/net_e_15_0_500x10.pkg', 'v1/mol_e_15_0_500x10.txt')
    # main('v1/net_e_20_0_500x10.pkg', 'v1/mol_e_20_0_500x10.txt')
    # main('v1/net_e_25_0_500x10.pkg', 'v1/mol_e_25_0_500x10.txt')
    #
    # main('v1/net_a_5_1_500x10.pkg', 'v1/mol_a_5_1_500x10.txt')
    # main('v1/net_a_10_1_500x10.pkg', 'v1/mol_a_10_1_500x10.txt')
    # main('v1/net_a_15_1_500x10.pkg', 'v1/mol_a_15_1_500x10.txt')
    # main('v1/net_a_20_1_500x10.pkg', 'v1/mol_a_20_1_500x10.txt')
    # main('v1/net_a_25_1_500x10.pkg', 'v1/mol_a_25_1_500x10.txt')
    #
    # main('v1/net_a_5_0_500x10.pkg', 'v1/mol_a_5_0_500x10.txt')
    # main('v1/net_a_10_0_500x10.pkg', 'v1/mol_a_10_0_500x10.txt')
    # main('v1/net_a_15_0_500x10.pkg', 'v1/mol_a_15_0_500x10.txt')
    # main('v1/net_a_20_0_500x10.pkg', 'v1/mol_a_20_0_500x10.txt')
    # main('v1/net_a_25_0_500x10.pkg', 'mol_a_25_0_500x10.txt')
    # main('v2/net_REINVENT_ex_ex.pkg', 'v2/mol_REINVENT_ex_ex.pkg')


if __name__ == '__main__':
    main()
