"""
training

Created by: Martin Sicho
On: 19-11-19, 15:23
"""
import torch

import os

from abc import ABC, abstractmethod

import torch as T
from sklearn.externals import joblib
from torch import Tensor

from drugex import model, util
from drugex.api.corpus import Corpus

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


class Generator(ABC):

    @staticmethod
    def load(deserializer : GeneratorDeserializer):
        return deserializer.getGenerator()

    def __init__(self, corpus : Corpus, train_params = None, generator_class = model.Generator):
        self.corpus = corpus
        self.model_class = generator_class
        self.model = self.model_class(self.corpus.voc)
        self.train_params = train_params if train_params else dict()

    @abstractmethod
    def pretrain(self, train_loader_params, validation_size=0, valid_loader_params=None):
        pass

    @abstractmethod
    def sample(self, n_samples, explore=None, epsilon=0.01, include_tensors=True, mc=1):
        pass

    @abstractmethod
    def policyUpdate(self, seq, pred):
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
            self.in_pickle_path = os.path.join(self.in_dir, 'net_{0}.pkg'.format(self.in_identifer))

        def getGenerator(self):
            ret = BasicGenerator(self.corpus, self.out_dir, self.out_identifer, self.train_params)
            ret.model.load_state_dict(T.load(self.in_pickle_path))
            return ret

    class BasicSerializer(GeneratorSerializer):

        def __init__(self, out_dir, out_identifier):
            self.out_dir = out_dir
            self.out_identifier = out_identifier
            self.out_pickle_path = os.path.join(out_dir, 'net_{0}.pkg'.format(out_identifier))

        def saveGenerator(self, generator):
            torch.save(generator.model.state_dict(), self.out_pickle_path)

    def __init__(self, corpus, out_dir, out_identifier="pr", train_params = None):
        super().__init__(corpus, train_params)

        # initialize output directories
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.net_pickle_path = os.path.join(self.out_dir, 'net_{0}.pkg'.format(out_identifier))
        self.net_log_path = os.path.join(self.out_dir, 'net_{0}.log'.format(out_identifier))

    def pretrain(self, train_loader_params, validation_size=0, valid_loader_params=None):
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

    def sample(self, n_samples, explore=None, mc=1, epsilon=0.01, include_tensors=False):
        seqs = []

        # repeated sampling with MC times
        for _ in range(mc):
            seq = self.model.sample(n_samples, explore=explore.model if explore else None, epsilon=epsilon)
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)
        ix = util.unique(seqs)
        seqs = seqs[ix]
        smiles, valids = util.check_smiles(seqs, self.corpus.voc)
        if include_tensors:
            return smiles, valids, seqs
        else:
            return smiles, valids

    def policyUpdate(self, seq : Tensor, pred : Tensor):
        score = self.model.likelihood(seq)
        self.model.optim.zero_grad()
        loss = self.model.PGLoss(score, pred)
        loss.backward()
        self.model.optim.step()