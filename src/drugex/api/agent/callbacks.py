"""
callbacks

Created by: Martin Sicho
On: 11/25/19, 10:50 AM
"""
import os
from abc import ABC, abstractmethod

import torch

from drugex.api.model.callbacks import GeneratorModelCallback
from drugex.api.pretrain.serialization import StateProvider


class AgentMonitor(GeneratorModelCallback, StateProvider):

    @abstractmethod
    def finalizeEpoch(self, current_epoch, total_epochs):
        pass

    @abstractmethod
    def smiles(self, smiles, score):
        pass

    @abstractmethod
    def performance(self, scores, valids, criterion, best_score):
        pass

    @abstractmethod
    def close(self):
        pass

class BasicAgentMonitor(AgentMonitor):

    def __init__(self, out_dir : str, identifier : str):
        self.out_dir = out_dir
        self.identifier = identifier
        self.log_file = open(os.path.join(self.out_dir, 'net_' + self.identifier + '.log'), 'w')
        self.net_pickle_path = os.path.join(self.out_dir, 'net_' + self.identifier + '.pkg')
        self.best_state = None
        self.mean_scores = []
        self.mean_valids = []
        self.criterions = []

    def finalizeEpoch(self, current_epoch, total_epochs):
        print("Epoch+: %d average: %.4f valid: %.4f unique: %.4f" % (current_epoch, self.mean_scores[-1], self.mean_valids[-1], self.criterions[-1]), file=self.log_file)
        self.log_file.flush()

    def performance(self, scores, valids, criterion, best_score):
        self.mean_scores.append(scores.mean())
        self.mean_valids.append(valids.mean())
        self.criterions.append(criterion)

    def smiles(self, smiles, score):
        print('%f\t%s' % (score, smiles), file=self.log_file)

    def state(self, current_state, is_best=False):
        if is_best:
            self.best_state = current_state
            torch.save(self.best_state, self.net_pickle_path)

    def model(self, model):
        pass

    def close(self):
        self.log_file.close()
        if self.best_state:
            torch.save(self.best_state, self.net_pickle_path)

    def getState(self):
        return self.best_state

