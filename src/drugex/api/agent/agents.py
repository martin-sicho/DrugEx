"""
agents

Created by: Martin Sicho
On: 21-11-19, 12:04
"""
import os

from abc import ABC, abstractmethod
from tqdm import trange

from drugex import util, VOC_DEFAULT
from drugex.api.agent.policy import PolicyGradient
from drugex.api.corpus import CorpusCSV
from drugex.api.environ.models import Environ
from drugex.api.pretrain.generators import Generator, BasicGenerator, GeneratorSerializer


class Agent(ABC):

    class UntrainedException(Exception):
        pass

    def __init__(self, environ : Environ, exploit : Generator, policy : PolicyGradient, serializer : GeneratorSerializer, explore = None, n_epochs=1000):
        self.environ = environ
        self.exploit = exploit
        self.explore = explore
        self.policy = policy
        self.n_epochs = n_epochs
        self.serializer = serializer
        self.optimal_generator = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def sample(self, n_samples):
        pass

    def saveOptimal(self):
        self.serializer.saveGenerator(self.optimal_generator)

class BasicDrugExAgent(Agent):

    def __init__(self, environ: Environ, exploit: Generator, policy: PolicyGradient, serializer : BasicGenerator.BasicSerializer, explore=None, n_epochs=1000):
        super().__init__(environ, exploit, policy, serializer, explore, n_epochs)
        self.serializer = serializer
        #: file path of hidden states of optimal exploitation network
        self.out_pickle_path = self.serializer.out_pickle_path

    def train(self):
        best_score = 0
        log_file = open(os.path.join(self.serializer.out_dir, 'net_' + self.serializer.out_identifier + '.log'), 'w')

        it = trange(self.n_epochs)
        for epoch in it:
            it.write('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            it.write('\nForward Policy Gradient Training Generator : ')
            self.policy(self.environ, self.exploit, explore=self.explore)

            # choosing the best model
            smiles, valids = self.exploit.sample(1000)
            scores = self.environ.predictSMILES(smiles)
            scores[valids == False] = 0
            unique = (scores >= 0.5).sum() / 1000
            # The model with best percentage of unique desired SMILES will be persisted on the hard drive.
            if best_score < unique:
                self.serializer.saveGenerator(self.exploit)
                best_score = unique
            print("Epoch+: %d average: %.4f valid: %.4f unique: %.4f" % (epoch, scores.mean(), valids.mean(), unique), file=log_file)
            for i, smile in enumerate(smiles):
                print('%f\t%s' % (scores[i], smile), file=log_file)

            # Learing rate exponential decay
            for param_group in self.exploit.model.optim.param_groups:
                param_group['lr'] *= (1 - 0.01)

        log_file.close()

        des = BasicGenerator.BasicDeserializer(CorpusCSV(VOC_DEFAULT), in_dir=self.serializer.out_dir, in_identifier=self.serializer.out_identifier)
        self.optimal_generator = des.getGenerator()

    def sample(self, n_samples):
        if self.optimal_generator:
            return self.optimal_generator.sample(n_samples=n_samples)
        else:
            raise self.UntrainedException("You have to train the agent first!")
