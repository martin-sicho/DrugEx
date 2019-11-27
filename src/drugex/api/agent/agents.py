"""
agents

Created by: Martin Sicho
On: 21-11-19, 12:04
"""

from abc import abstractmethod
from tqdm import trange

from drugex.api.agent.callbacks import AgentMonitor
from drugex.api.agent.policy import PolicyGradient
from drugex.api.environ.models import Environ
from drugex.api.pretrain.generators import Generator
from drugex.api.pretrain.serialization import StateProvider


class AgentTrainer(StateProvider):

    class UntrainedException(Exception):
        pass

    def __init__(self, monitor : AgentMonitor, environ : Environ, exploit : Generator, policy : PolicyGradient, explore = None, train_params=None):
        self.monitor = monitor
        self.environ = environ
        self.exploit = exploit
        self.explore = explore
        self.policy = policy
        self.train_params = train_params if train_params else dict()
        if "n_epochs" not in self.train_params:
            self.train_params.update({"n_epochs" : 1000})
        self.n_epochs = self.train_params["n_epochs"]

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def sample(self, n_samples):
        pass

    @abstractmethod
    def getAgent(self):
        pass

class DrugExAgentTrainer(AgentTrainer):

    def __init__(self, monitor : AgentMonitor, environ: Environ, exploit: Generator, policy: PolicyGradient, explore=None, train_params=None):
        super().__init__(monitor, environ, exploit, policy, explore, train_params=train_params)
        self.best_state = None

    def getState(self):
        return self.best_state

    def getAgent(self):
        self.exploit.setState(self.best_state)
        return self.exploit

    def train(self):
        best_score = 0
        for epoch in trange(self.n_epochs, desc="Epoch"):
            self.policy(self.environ, self.exploit, explore=self.explore)
            self.monitor.model(self.exploit.model)

            # choosing the best model
            smiles, valids = self.exploit.sample(1000)
            scores = self.environ.predictSMILES(smiles)
            scores[valids == False] = 0
            unique = (scores >= 0.5).sum() / 1000

            # The model with best percentage of unique desired SMILES will be persisted on the hard drive.
            is_best = False
            if best_score < unique or self.best_state is None:
                is_best = True
                self.best_state = self.exploit.getState()
                best_score = unique

            # monitor performance information
            self.monitor.performance(scores, valids, unique, best_score)
            for i, smile in enumerate(smiles):
                self.monitor.smiles(smile, scores[i])

            # monitor state
            self.monitor.state(self.exploit.getState(), is_best)

            # Learning rate exponential decay
            for param_group in self.exploit.model.optim.param_groups:
                param_group['lr'] *= (1 - 0.01)

            # finalize epoch monitoring
            self.monitor.finalizeEpoch(epoch, self.n_epochs)

        self.monitor.close()
        self.exploit.setState(self.best_state)

    def sample(self, n_samples):
        if self.best_state:
            return self.exploit.sample(n_samples=n_samples)
        else:
            raise self.UntrainedException("You have to train the agent first!")
