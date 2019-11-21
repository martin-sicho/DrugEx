"""
models

Created by: Martin Sicho
On: 19-11-19, 9:59
"""
import os

from abc import ABC, abstractmethod
from sklearn.externals import joblib

from drugex.api.environ.data import EnvironData

class EnvironSerializer(ABC):

    @abstractmethod
    def saveModel(self, model):
         pass

class EnvironDeserializer(ABC):

    @abstractmethod
    def getModel(self):
         pass

class Environ(ABC):

    @staticmethod
    def load(deserializer : EnvironDeserializer):
        return deserializer.getModel()

    def __init__(self, train_provider : EnvironData, test_provider=None):
        self.train_provider = train_provider
        self.test_provider = test_provider

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predictSMILES(self, smiles):
        pass

    @abstractmethod
    def getPerfData(self):
        pass

    def save(self, serializer : EnvironSerializer):
        serializer.saveModel(self)

class FileEnvSerializer(EnvironSerializer):

    def __init__(self, out_dir, identifier="EnvironModel", include_perf=True):
        self.identifier = identifier
        self.include_perf = include_perf
        self.out_dir = out_dir

    def saveModel(self, model : Environ):
        joblib.dump(model, os.path.join(self.out_dir, "{0}.pkg".format(self.identifier)))

        if self.include_perf:
            cv, test = model.getPerfData()
            cv.to_csv(os.path.join(self.out_dir, "{0}.cv.txt".format(self.identifier)), index=None)
            test.to_csv(os.path.join(self.out_dir, "{0}.ind.txt".format(self.identifier)), index=None)

class FileEnvDeserializer(EnvironDeserializer):

    def __init__(self, path):
        self.path = path

    def getModel(self):
        return joblib.load(self.path)


