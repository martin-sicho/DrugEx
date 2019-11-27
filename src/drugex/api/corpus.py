"""
data

Created by: Martin Sicho
On: 15-11-19, 15:10
"""
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from drugex import VOC_DEFAULT
from drugex.core import util
from drugex.core.util import Voc

class Corpus(ABC):

    class InvalidOperation(Exception):
        pass

    def __init__(self, vocabulary = VOC_DEFAULT):
        self.voc = vocabulary
        self.words = set(self.voc.chars)
        self.sep = "\t"

class BasicCorpus(Corpus):
    pass

class DataProvidingCorpus(Corpus):

    SUB_RE = re.compile(r'\[\d+')

    def __init__(self, vocabulary = VOC_DEFAULT, token="SENT", smiles_col="CANONICAL_SMILES"):
        super().__init__(vocabulary)
        self.token = token
        self.sampled_idx = None
        self.df = pd.DataFrame()
        self.smiles_column = smiles_col

    @abstractmethod
    def updateData(self, update_voc=False, sample=None):
        pass

    def saveVoc(self, out):
        # persisting the vocabulary on the hard drive.
        with open(out, 'w') as file:
            file.write('\n'.join(sorted(self.words)))

    def saveCorpus(self, out):
        self.df.to_csv(out, sep=self.sep, index=None)

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

        if not loader_params:
            loader_params = {
                "batch_size" : 512
                , "shuffle" : True
                , "drop_last" : False
                , "collate_fn" : util.MolData.collate_fn
            }

        if self.df.empty:
            raise self.InvalidOperation("Corpus data is empty. Update it before extracting a data loader.")

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
            return DataLoader(sample, **loader_params)
        else:
            return DataLoader(sample)

    def resetDF(self, canons, tokens, update_voc):
        self.df = pd.DataFrame()

        # saving the canonical smiles and token sentences as a basis for future transformations
        self.df[self.smiles_column] = canons
        self.df[self.token] = tokens
        self.df.drop_duplicates(subset=self.smiles_column, inplace=True)

        # rewrite the current voc instance if requested
        if update_voc:
            self.voc = Voc(chars=self.words)

    @staticmethod
    def fromDataFrame(df, voc, smiles_column, sample=None):
        df = df[smiles_column].dropna().drop_duplicates()
        tokens = []
        canons = []
        words = set()
        if sample:
            df = df.sample(sample)
        smiles = set()
        it = tqdm(df, desc='Reading SMILES')
        for smile in it:
            # replacing the radioactive atom into nonradioactive atom
            smile = DataProvidingCorpus.SUB_RE.sub('[', smile)
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
                token = voc.tokenize(smile)
                if len(token) <= 100:
                    words.update(token)
                    canons.append(Chem.CanonSmiles(smile, 0))
                    tokens.append(' '.join(token))
            except Exception as e:
                it.write('{} {}'.format(e, smile))

        return tokens, canons, words

class CorpusCSV(DataProvidingCorpus):

    @staticmethod
    def fromFiles(
            corpus_path
            , vocab_path=None
            , smiles_column='CANONICAL_SMILES'
            , sep='\t'
            , token="SENT"
            , n_rows=None
    ):
        ret = CorpusCSV(corpus_path, Voc(vocab_path) if vocab_path else VOC_DEFAULT, token=token, smiles_column=smiles_column, sep=sep)
        ret.df = pd.read_table(corpus_path) if not n_rows else pd.read_table(corpus_path, nrows=n_rows)
        return ret

    def __init__(self, update_file : str, vocabulary = VOC_DEFAULT, smiles_column='CANONICAL_SMILES', sep='\t', token="SENT"):
        super().__init__(vocabulary, token, smiles_col=smiles_column)
        self.sep = sep
        self.update_file = update_file

    def updateData(self, update_voc = False, sample=None):
        """Constructing the molecular corpus by splitting each SMILES into
        a range of tokens contained in vocabulary.

        Arguments:
            input_file : the path of tab-delimited data file that contains CANONICAL_SMILES.
            out : the path for vocabulary (containing all of tokens for SMILES construction)
                and output table (including CANONICAL_SMILES and whitespace delimited token sentence)
        """
        self.words = set()

        df = pd.read_table(self.update_file, sep=self.sep)
        tokens, canons, self.words = self.fromDataFrame(df, self.voc, self.smiles_column, sample=sample)

        self.resetDF(canons, tokens, update_voc)

        return self.df, self.voc

class CorpusChEMBL(DataProvidingCorpus):

    def __init__(self
                 , gene_names : list
                 , clean_raw=False
                 , smiles_field="CANONICAL_SMILES"
                 , extracted_fields=(
                    "MOLECULE_CHEMBL_ID"
                    , "CANONICAL_SMILES"
                    , "PCHEMBL_VALUE"
                    , "ACTIVITY_COMMENT"
                    )
                 ):
        super().__init__(smiles_col=smiles_field)
        self.gene_names = gene_names
        self.extracted_fields = extracted_fields
        self.CHEMBL_TARGETS = new_client.target
        self.CHEMBL_COMPOUNDS = new_client.molecule
        self.CHEMBL_ACTIVITIES = new_client.activity
        self.raw_data = pd.DataFrame()
        self.clean_raw = clean_raw

    def _cleanRaw(self):
        subset = set(self.extracted_fields)
        subset.discard("ACTIVITY_COMMENT")
        self.raw_data = self.raw_data.dropna(subset=subset)
        for i, row in self.raw_data.iterrows():
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
            self.raw_data.loc[i, 'CANONICAL_SMILES'] = Chem.CanonSmiles(smile, 0)
            # removing molecule contained metal atom
            if '[Au]' in smile or '[As]' in smile or '[Hg]' in smile or '[Se]' in smile or smile.count('C') + smile.count('c') < 2:
                self.raw_data = self.raw_data.drop(i)
        self.raw_data = self.raw_data.sample(frac=1)

    def updateData(self, update_voc=False, sample=None):
        self.words = set()
        self.raw_data = pd.DataFrame()

        for gene_name in self.gene_names:
            target_chembl_ids = []
            for result in self.CHEMBL_TARGETS.filter(
                    target_synonym__icontains=gene_name
                    , target_type='SINGLE PROTEIN'
                    , organism='Homo sapiens'
            ):
                target_chembl_ids.append(result['target_chembl_id'])
            print("Found following target chembl IDs related to {0}".format(gene_name), target_chembl_ids)

            compound_data = dict()
            for field in self.extracted_fields:
                compound_data[field] = []
            for result in tqdm(self.CHEMBL_ACTIVITIES.filter(
                    target_chembl_id__in=target_chembl_ids
                    # , pchembl_value__isnull=False
            ), desc="Downloading compound data..."):
                for field in self.extracted_fields:
                    compound_data[field].append(result[field.lower()])
            self.raw_data = self.raw_data.append(pd.DataFrame(compound_data))

        # cleanup the raw data in this instance
        if self.clean_raw:
            self._cleanRaw()

        tokens, canons, self.words = self.fromDataFrame(self.raw_data, self.voc, self.smiles_column, sample)

        self.resetDF(canons, tokens, update_voc)

        return self.df, self.voc