"""
serialization

Created by: Martin Sicho
On: 25-11-19, 14:45
"""
import os
from abc import ABC, abstractmethod
from sklearn.externals import joblib


class EnvironSerializer(ABC):

    @abstractmethod
    def saveModel(self, model):
         pass


class EnvironDeserializer(ABC):

    @abstractmethod
    def getModel(self):
         pass


class FileEnvSerializer(EnvironSerializer):

    def __init__(self, out_dir, identifier="EnvironModel", include_perf=True):
        self.identifier = identifier
        self.include_perf = include_perf
        self.out_dir = out_dir

    def saveModel(self, model):
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