"""
training

Created by: Martin Sicho
On: 19-11-19, 15:23
"""
import torch

from abc import abstractmethod

from torch import Tensor

from drugex.core import model, util
from drugex.api.corpus import Corpus, BasicCorpus
from drugex.api.pretrain.serialization import GeneratorDeserializer, StateProvider, GeneratorSerializer


class Generator(StateProvider):

    @staticmethod
    def load(deserializer : GeneratorDeserializer):
        return deserializer.getGenerator()

    def __init__(self, corpus : Corpus, initial_state = None):
        self.corpus = corpus
        self.model = model.Generator(self.corpus.voc, None)
        if initial_state:
            self.setState(initial_state)

    @abstractmethod
    def sample(self, n_samples, explore=None, epsilon=0.01, include_tensors=True, mc=1):
        pass

    @abstractmethod
    def setState(self, state : StateProvider):
        pass

    def save(self, serializer : GeneratorSerializer):
        serializer.saveGenerator(self)

class PretrainableGenerator(Generator):

    class PretrainingError(Exception):
        pass

    def __init__(self, corpus: Corpus, monitor=None, initial_state=None, train_params=None):
        super().__init__(corpus, initial_state)
        self.train_params = train_params if train_params else dict()
        if monitor:
            self.model.registerMonitor(monitor)

    @abstractmethod
    def pretrain(self, train_loader_params, validation_size=0, valid_loader_params=None):
        pass

class PolicyAwareGenerator(Generator):

    def __init__(self, corpus: Corpus, initial_state=None):
        super().__init__(corpus, initial_state)

    @abstractmethod
    def policyUpdate(self, seq, pred):
        pass

class BasicGenerator(PretrainableGenerator, PolicyAwareGenerator):

    def __init__(self, corpus = BasicCorpus(), monitor = None, initial_state = None, train_params = None):
        super().__init__(corpus, monitor, initial_state, train_params=train_params)
        self.monitor = monitor

    def setState(self, state : StateProvider):
        state_ = state.getState()
        if state:
            print("Loading initial state...")
            print("State provider:", state.__class__)
            self.model.load_state_dict(state_)
            print("Done")
        else:
            raise Exception("Empty state was provided.") # TODO: make this a custom exception

    def getState(self):
        return self.model.state_dict()

    def pretrain(self, train_loader_params, validation_size=0, valid_loader_params=None):
        if hasattr(self.corpus, "getDataLoader"):
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
                , **self.train_params
            )
        else:
            raise self.PretrainingError("The current corpus does not provide data loaders.")

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