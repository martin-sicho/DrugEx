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
from torch.utils.data import DataLoader
from tqdm import tqdm

from drugex import util
from drugex.util import Voc

class Corpus(ABC):

    def __init__(self, vocabulary : Voc, data_loader_cls = DataLoader):
        self.voc = vocabulary
        self.words = set(self.voc.chars)
        self.sub_re = re.compile(r'\[\d+')
        self.loader_cls = data_loader_cls

    @abstractmethod
    def getDataLoader(self, sample_size=None, exclude_sampled=True, loader_params=None):
        """
        Copnverts the underlying corpus data to a `DataLoader` instance.
        A subsample of the original data can be requested by specifying
        `sample_size`. The `exclude_sampled` can be used to keep
        track of the sampled data and exclude it in a future request.
        The exculded data points are returned back to the pool when
        they were ignored exactly once.

        :param sample_size: size of a sample (if None whole dataset is returned)
        :param exclude_sampled: if this is True and `sample_size` is specified, the sampled data will be excluded the next time we return a data loader
        :param loader_params: parameters for the data loader as a dictionary
        :return: a data loader to use in training
        """

        pass

class CorpusCSV(Corpus):

    @staticmethod
    def fromFiles(
            corpus_path
            , vocab_path
            , smiles_column='CANONICAL_SMILES'
            , sep='\t'
            , token="SENT"
            , n_rows=None
    ):
        ret = CorpusCSV(Voc(vocab_path), token=token, smiles_column=smiles_column, sep=sep)
        ret.df = pd.read_table(corpus_path) if not n_rows else pd.read_table(corpus_path, nrows=n_rows)
        return ret

    def __init__(self, vocabulary : Voc, data_loader_cls = DataLoader, smiles_column='CANONICAL_SMILES', sep='\t', token="SENT"):
        super().__init__(vocabulary=vocabulary, data_loader_cls=data_loader_cls)
        self.smiles_column = smiles_column
        self.token = token
        self.sep = sep
        self.df = pd.DataFrame()
        self.sampled_idx = None

    def update(self, input_file):
        """Constructing the molecular corpus by splitting each SMILES into
        a range of tokens contained in vocabulary.

        Arguments:
            input_file : the path of tab-delimited data file that contains CANONICAL_SMILES.
            out : the path for vocabulary (containing all of tokens for SMILES construction)
                and output table (including CANONICAL_SMILES and whitespace delimited token sentence)
        """
        tokens = []
        canons = []
        self.words = set()
        self.df = pd.DataFrame()

        df = pd.read_table(input_file, sep=self.sep)[self.smiles_column].dropna().drop_duplicates()
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
                    canons.append(Chem.CanonSmiles(smile, 0))
                    tokens.append(' '.join(token))
            except Exception as e:
                it.write('{} {}'.format(e, smile))

        # saving the canonical smiles and token sentences as a basis for future transformations
        self.df[self.smiles_column] = canons
        self.df[self.token] = tokens
        self.df.drop_duplicates(subset=self.smiles_column)

    def saveVoc(self, out):
        # persisting the vocabulary on the hard drive.
        with open(out, 'w') as file:
            file.write('\n'.join(sorted(self.words)))

    def saveCorpus(self, out):
        self.df.to_csv(out, sep=self.sep, index=None)

    def getDataLoader(self, sample_size=None, exclude_sampled=True, loader_params=None):
        if self.loader_cls == DataLoader:
            if sample_size is None:
                sample = self.df.drop(self.sampled_idx) if self.sampled_idx is not None else self.df
                self.sampled_idx = None
            else:
                sample = self.df.drop(self.sampled_idx).sample(sample_size) if self.sampled_idx else self.df.sample(sample_size)
                self.sampled_idx = None
                if exclude_sampled:
                    self.sampled_idx = sample.index
            sample = util.MolData(sample, self.voc, token='SENT')
            if loader_params:
                return self.loader_cls(sample, **loader_params)
            else:
                return self.loader_cls(sample)
        else:
            raise NotImplementedError("Uninplemented data loader requested: {0}".format(self.loader_cls))