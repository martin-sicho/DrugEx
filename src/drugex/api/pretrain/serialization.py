"""
serialization

Created by: Martin Sicho
On: 11/25/19, 10:29 AM
"""

import os
from sklearn.externals import joblib
from abc import ABC, abstractmethod

class GeneratorSerializer(ABC):

    @abstractmethod
    def saveGenerator(self, generator):
        pass

class GeneratorDeserializer:

    @abstractmethod
    def getGenerator(self):
        pass

class FileSerializer(GeneratorSerializer):

    def __init__(self, out_dir, identifier):
        self.out_dir = out_dir
        self.identifier = identifier

    def saveGenerator(self, generator):
        joblib.dump(generator, os.path.join(self.out_dir, "{0}.pkg".format(self.identifier)))

class FileDeserializer(GeneratorDeserializer):

    def __init__(self, in_dir, identifier):
        self.in_dir = in_dir
        self.identifier = identifier

    def getGenerator(self):
        joblib.load(os.path.join(self.in_dir, "{0}.pkg".format(self.identifier)))

# initial state provider

class StateProvider(ABC):

    @abstractmethod
    def getState(self):
        pass

