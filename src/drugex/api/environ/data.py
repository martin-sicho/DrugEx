"""
data

Created by: Martin Sicho
On: 19-11-19, 9:57
"""

from abc import ABC, abstractmethod

import pandas as pd

from drugex.core import util


class EnvironData(ABC):

    def __init__(self, is_regression=False, subsample_size=None):
        self.is_regression = is_regression
        self.subsample_size = subsample_size

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def getX(self):
        pass

    @abstractmethod
    def gety(self):
        pass

    @abstractmethod
    def getGroundTruthData(self):
        pass


class ChEMBLCSV(EnvironData):

    def __init__(self, input_file : str, activity_threshold=None, subsample_size=None, id_col='CMPD_CHEMBLID', is_regression=False):
        super().__init__(subsample_size=subsample_size, is_regression=is_regression)
        self.PAIR = [id_col, 'CANONICAL_SMILES', 'PCHEMBL_VALUE', 'ACTIVITY_COMMENT']
        self.activity_threshold = activity_threshold
        if self.activity_threshold is not None:
            self.is_regression = False
        else:
            self.is_regression = True
        self.input_file = input_file
        self.df = pd.DataFrame()
        self.X = None
        self.y = None
        self.update()

    def getSMILES(self):
        return self.df.CANONICAL_SMILES

    def update(self):
        if self.subsample_size is not None:
            self.df = pd.read_table(self.input_file).sample(self.subsample_size)
        else:
            self.df = pd.read_table(self.input_file)
        self.df = self.df[self.PAIR].set_index(self.PAIR[0])
        self.df[self.PAIR[2]] = self.df.groupby(self.df.index).mean()

        # The molecules that have PChEMBL value
        numery = self.df[self.PAIR[1:-1]].drop_duplicates().dropna()
        if self.is_regression:
            self.df = numery
            self.y = numery[self.PAIR[2:3]].values
        else:
            # The molecules that do not have PChEMBL value
            # but has activity comment to show whether it is active or not.
            binary = self.df[self.df.ACTIVITY_COMMENT.str.contains('Active') == True].drop_duplicates()
            binary = binary[~binary.index.isin(numery.index)]
            # binary.loc[binary.ACTIVITY_COMMENT == 'Active', 'PCHEMBL_VALUE'] = 100.0
            binary.loc[binary.ACTIVITY_COMMENT.str.contains('Not'), 'PCHEMBL_VALUE'] = 0.0
            binary = binary[self.PAIR[1:3]].dropna().drop_duplicates()
            self.df = numery.append(binary)
            # For is_regression model the active ligand is defined as
            # PChBMBL value >= activity_threshold
            self.y = (self.df[self.PAIR[2:3]] >= self.activity_threshold).astype(float).values

        # ECFP6 fingerprints extraction
        self.X = util.Environment.ECFP_from_SMILES(self.df.CANONICAL_SMILES).values
    def getX(self):
        return self.X

    def gety(self):
        return self.y

    def getGroundTruthData(self):
        data = pd.DataFrame()
        data['CANONICAL_SMILES'], data['LABEL'] = self.getSMILES(), self.gety()[:, 0]
        return data