"""
data

Created by: Martin Sicho
On: 19-11-19, 9:57
"""

from abc import ABC, abstractmethod


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
