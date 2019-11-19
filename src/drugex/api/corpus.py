"""
data

Created by: Martin Sicho
On: 15-11-19, 15:10
"""
import re
from abc import ABC, abstractmethod

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