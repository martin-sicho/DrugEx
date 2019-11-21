"""
training

Created by: Martin Sicho
On: 19-11-19, 15:23
"""
import os

from abc import ABC, abstractmethod

import torch as T

from drugex import model
from drugex.api.corpus import Corpus

class GeneratorSerializer(ABC):

    @abstractmethod
    def saveGenerator(self, generator):
        pass

class GeneratorDeserializer:

    @abstractmethod
    def getGenerator(self):
        pass

class Generator(ABC):

    @staticmethod
    def load(deserializer : GeneratorDeserializer):
        return deserializer.getGenerator()

    def __init__(self, corpus : Corpus, train_params = None, generator_class = model.Generator):
        self.corpus = corpus
        self.generator_class = generator_class
        self.model = self.generator_class(self.corpus.voc)
        self.train_params = train_params if train_params else dict()

    @abstractmethod
    def train(self, train_loader_params, validation_size=0, valid_loader_params=None):
        pass

    @abstractmethod
    def generate(self, samples):
        pass

    def save(self, serializer : GeneratorSerializer):
        serializer.saveGenerator(self)

class BasicGenerator(Generator):

    class BasicDeserializer(GeneratorDeserializer):

        def __init__(self, corpus : Corpus, in_dir, in_identifier, out_dir=None, out_identifier=None, train_params=None):
            self.corpus = corpus
            self.out_dir = out_dir if out_dir else in_dir
            self.in_dir = in_dir
            self.out_identifer = out_identifier if out_identifier else in_identifier
            self.in_identifer = in_identifier
            self.train_params = train_params
            self.in_pickle_path = os.path.join(os.path.join(self.in_dir, 'net_{0}.pkg'.format(self.in_identifer)))

        def getGenerator(self):
            ret = BasicGenerator(self.corpus, self.out_dir, self.out_identifer, self.train_params)
            ret.model.load_state_dict(T.load(self.in_pickle_path))
            return ret

    def __init__(self, corpus, out_dir, out_identifier="pr", train_params = None):
        super().__init__(corpus, train_params)

        # initialize output directories
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.net_pickle_path = os.path.join(self.out_dir, 'net_{0}.pkg'.format(out_identifier))
        self.net_log_path = os.path.join(self.out_dir, 'net_{0}.log'.format(out_identifier))

    def train(self, train_loader_params, validation_size=0, valid_loader_params=None):
        if validation_size > 0:
            valid_loader = self.corpus.getDataLoader(
                loader_params=valid_loader_params
                , sample_size=validation_size
                , exclude_sampled=True
            )
            train_loader = self.corpus.getDataLoader(loader_params=train_loader_params)
        else:
            train_loader = self.corpus.getDataLoader(loader_params=train_loader_params)
            valid_loader = None

        self.model.fit(
            train_loader
            , loader_valid=valid_loader
            , out_path=self.net_pickle_path
            , log_path=self.net_log_path
            , **self.train_params
        )

    def generate(self, samples):
        raise NotImplementedError