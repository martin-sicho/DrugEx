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


class Pretrainer(ABC):

    def __init__(self, corpus : Corpus, train_params = None, initial_state = None, generator_class = model.Generator):
        self.corpus = corpus
        self.generator_class = generator_class
        self.generator = self.generator_class(self.corpus.voc)
        if initial_state:
            self.generator.load_state_dict(T.load(initial_state))
        self.train_params = train_params

    @abstractmethod
    def train(self, train_loader_params, validation_size=0, valid_loader_params=None):
        pass

class LoggingPretrainer(Pretrainer):

    def __init__(self, corpus, out_dir, out_identifier, train_params = None, initial_state = None):
        super().__init__(corpus, train_params, initial_state)
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

        self.generator.fit(
            train_loader
            , loader_valid=valid_loader
            , out_path=self.net_pickle_path
            , log_path=self.net_log_path
            , **self.train_params
        )