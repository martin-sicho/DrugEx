"""
models

Created by: Martin Sicho
On: 19-11-19, 9:59
"""

from abc import ABC, abstractmethod

class EnvironModel(ABC):

    def __init__(self, train_provider, test_provider=None):
        self.train_provider = train_provider
        self.test_provider = test_provider
        self.model = None

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def getPerfData(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass

