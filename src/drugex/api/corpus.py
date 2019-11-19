"""
data

Created by: Martin Sicho
On: 15-11-19, 15:10
"""
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from drugex.util import Voc

class Corpus(ABC):

    def __init__(self, vocabulary : Voc):
        self.voc = vocabulary
        self.words = set()
        self.canons = []
        self.tokens = []
        self.sub_re = re.compile(r'\[\d+')

    @abstractmethod
    def getMolData(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def saveVoc(self, filename):
        pass

    @abstractmethod
    def saveCorpus(self, filename):
        pass

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
        raise NotImplementedError # TODO: implement