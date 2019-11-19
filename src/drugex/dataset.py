#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file is used for dataset construction.

It contains two dataset as follows:

1. ZINC set: it is used for pre-training model
2. A2AR set: it is used for fine-tuning model and training predictor
"""

import os
import re

import click
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from drugex.api.corpus import Corpus
from drugex.util import Voc

class CorpusCSV(Corpus):

    def __init__(self, input_file: str, vocab_path : str, smiles_column='CANONICAL_SMILES', sep='\t'):
        super().__init__(vocabulary=Voc(vocab_path))
        self.input_file = input_file
        self.smiles_column = smiles_column
        self.sep = sep

    def update(self):
        """Constructing the molecular corpus by splitting each SMILES into
        a range of tokens contained in vocabulary.

        Arguments:
            input : the path of tab-delimited data file that contains CANONICAL_SMILES.
            out : the path for vocabulary (containing all of tokens for SMILES construction)
                and output table (including CANONICAL_SMILES and whitespace delimited token sentence)
        """
        self.tokens = []
        self.canons = []
        self.words = set()

        df = pd.read_table(self.input_file, sep=self.sep)[self.smiles_column].dropna().drop_duplicates()
        smiles = set()
        it = tqdm(df, desc='Reading SMILES')
        for smile in it:
            # replacing the radioactive atom into nonradioactive atom
            smile = self.sub_re.sub('[', smile)
            # reserving the largest one if the molecule contains more than one fragments,
            # which are separated by '.'.
            if '.' in smile:
                frags = smile.split('.')
                ix = np.argmax([len(frag) for frag in frags])
                smile = frags[ix]
                # TODO replace with: smile = max(frags, key=len)
            # if it doesn't contain carbon atom, it cannot be drug-like molecule, just remove
            if smile.count('C') + smile.count('c') < 2:
                continue
            if smile in smiles:
                it.write('duplicate: {}'.format(smile))
            smiles.add(smile)
        # collecting all of the tokens in the sentences for vocabulary construction.
        it = tqdm(smiles, desc='Collecting tokens')
        for smile in it:
            try:
                token = self.voc.tokenize(smile)
                if len(token) <= 100:
                    self.words.update(token)
                    self.canons.append(Chem.CanonSmiles(smile, 0))
                    self.tokens.append(' '.join(token))
            except Exception as e:
                it.write('{} {}'.format(e, smile))

    def saveVoc(self, out):
        # persisting the vocabulary on the hard drive.
        with open(out, 'w') as file:
            file.write('\n'.join(sorted(self.words)))

    def saveCorpus(self, out):
        # saving the canonical smiles and token sentences as a table into hard drive.
        log = pd.DataFrame()
        log['CANONICAL_SMILES'] = self.canons
        log['SENT'] = self.tokens
        log.drop_duplicates(subset='CANONICAL_SMILES')
        log.to_csv(out, sep='\t', index=None)

    def getMolData(self):
        pass # TODO: implement


def ZINC(folder: str, out: str):
    """Uniformly random selecting molecule from ZINC database for Construction of ZINC set,
    which is used for pre-trained model training.

    Arguments:
        folder : the directory of the ZINC database, it contains all of the molecules
            that are separated into different files based on the logP and molecular weight.
        out : the file path of output dataframe, it contains all of randomly selected molecules,
            also including its SMILES string, logP and molecular weight
    """
    files = os.listdir(folder)
    points = [(i, j) for i in range(200, 600, 25) for j in np.arange(-2, 6, 0.5)]
    select = pd.DataFrame()
    for symbol in tqdm([i+j for i in 'ABCDEFGHIJK' for j in 'ABCDEFGHIJK']):
        zinc = pd.DataFrame()
        for fname in files:
            if not fname.endswith('.txt'): continue
            if not fname.startswith(symbol): continue
            df = pd.read_table(folder+fname)[['mwt', 'logp', 'smiles']]
            df.columns = ['MWT', 'LOGP', 'CANONICAL_SMILES']
            zinc = zinc.append(df)
        for mwt, logp in points:
            df = zinc[(zinc.MWT > mwt) & (zinc.MWT <= (mwt + 25))]
            df = df[(df.LOGP > logp) & (df.LOGP <= (logp+0.5))]
            if len(df) > 2500:
                df = df.sample(2500)
            select = select.append(df)
    select.to_csv(out, sep='\t', index=None)


def A2AR(input_path: str, output_path: str):
    """Construction of A2AR set, which is used for fine-tuned model and predictor training.
    Arguments:
        input_path : the path of tab-delimited data file that contains CANONICAL_SMILES.
        output_path : the path saving the refined data after filtering the invalid data,
            including removing molecule contained metal atom, reserving the largest fragments,
            and replacing the nitrogen electrical group to nitrogen atom "N".
    """
    df = pd.read_table(input_path)
    PAIR = ['CMPD_CHEMBLID', 'CANONICAL_SMILES', 'PCHEMBL_VALUE', 'ACTIVITY_COMMENT']
    df = df[PAIR]
    df = df.dropna(subset=PAIR[:-1])
    for i, row in df.iterrows():
        # replacing the nitrogen electrical group to nitrogen atom "N"
        smile = row['CANONICAL_SMILES'].replace('[NH+]', 'N').replace('[NH2+]', 'N').replace('[NH3+]', 'N')
        # removing the radioactivity of each atom
        smile = re.sub('\[\d+', '[', smile)
        # reserving the largest fragments
        if '.' in smile:
            frags = smile.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smile = frags[ix]
        # Transforming into canonical SMILES based on the Rdkit built-in algorithm.
        df.loc[i, 'CANONICAL_SMILES'] = Chem.CanonSmiles(smile, 0)
        # removing molecule contained metal atom
        if '[Au]' in smile or '[As]' in smile or '[Hg]' in smile or '[Se]' in smile or smile.count('C') + smile.count('c') < 2:
            df = df.drop(i)
    # df = df.drop_duplicates(subset='CANONICAL_SMILES')
    df.to_csv(output_path, index=False, sep='\t')


@click.command()
@click.option('-d', '--data-directory', type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.option('-e', '--environment-data-file', type=click.Path(dir_okay=False, file_okay=True), required=True)
@click.option('-v', '--vocabulary-file', default='voc.txt', type=click.Path(dir_okay=False, file_okay=True))
def main(data_directory, environment_data_file, vocabulary_file):
    zinc_output_path = os.path.join(data_directory, 'ZINC.txt')
    zinc_directory = os.path.join(data_directory, 'zinc')
    if os.path.exists(zinc_directory):
        click.echo("Compiling {0}".format(zinc_output_path))
        ZINC(folder=zinc_directory, out=zinc_output_path)
        click.echo("Done.")
    else:
        click.echo('Missing ZINC folder: {0} \nThe default ZINC output file will be used: {1}'.format(zinc_directory, zinc_output_path), err=True)

    click.echo("Generating ZINC corpus from: {0}".format(zinc_output_path))
    zinc_processed = os.path.exists(os.path.join(data_directory, 'zinc_corpus.txt')) \
                 and os.path.exists(os.path.join(data_directory, 'zinc_voc.txt'))
    if os.path.exists(zinc_output_path) and not zinc_processed:
        corpus = CorpusCSV(zinc_output_path, vocab_path=os.path.join(data_directory, vocabulary_file))
        corpus.update()
        corpus.saveVoc(os.path.join(data_directory, 'zinc_voc.txt'))
        corpus.saveCorpus(os.path.join(data_directory, 'zinc_corpus.txt'))
        click.echo("Done.")
    elif zinc_processed:
        click.echo('ZINC data was already processed (corpus and vocabulary files generated). Skipping...', err=True)
        pass
    else:
        raise FileNotFoundError('Missing ZINC output file: {}'.format(zinc_output_path))

    environment_data_file = os.path.join(data_directory, environment_data_file)
    click.echo("Generating ChEMBL corpus from: {0}".format(environment_data_file))
    chembl_processed = os.path.exists(os.path.join(data_directory, 'chembl_corpus.txt')) \
                 and os.path.exists(os.path.join(data_directory, 'chembl_voc.txt'))
    if os.path.exists(environment_data_file) and not chembl_processed:
        corpus = CorpusCSV(environment_data_file, vocab_path=os.path.join(data_directory, vocabulary_file))
        corpus.update()
        corpus.saveVoc(os.path.join(data_directory, 'chembl_voc.txt'))
        corpus.saveCorpus(os.path.join(data_directory, 'chembl_corpus.txt'))
        click.echo("Done.")
    elif chembl_processed:
        click.echo('ChEMBL data was already processed (corpus and vocabulary files generated). Skipping...', err=True)
        pass
    else:
        raise FileNotFoundError('Missing ChEMBL output file: {}'.format(environment_data_file))

    click.echo("Parsing environment CHEMBL data at {0}".format(environment_data_file))
    output_path = os.path.join(data_directory, 'FT_ENV_data.txt')
    if os.path.exists(environment_data_file):
        A2AR(environment_data_file, output_path)
        click.echo("Done.")
    else:
        raise FileNotFoundError('Missing environment data path: {0}'.format(environment_data_file))

if __name__ == '__main__':
    main()
