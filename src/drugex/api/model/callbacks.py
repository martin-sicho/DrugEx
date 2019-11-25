"""
loggers

Created by: Martin Sicho
On: 11/24/19, 7:05 PM
"""
import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from drugex.api.pretrain.serialization import StateProvider


class GeneratorModelCallback(ABC):

    @abstractmethod
    def model(self, model):
        pass

    @abstractmethod
    def state(self, current_state, is_best=False):
        pass

    @abstractmethod
    def close(self):
        pass

class PretrainingMonitor(GeneratorModelCallback, StateProvider):

    @abstractmethod
    def finalizeStep(
            self
            , current_epoch : int
            , current_step : int
            , total_epochs : int
            , total_steps : int
    ):
        pass

    @abstractmethod
    def performance(self, loss_train, loss_valid, error_rate, best_error):
        pass

    @abstractmethod
    def smiles(self, smiles, score):
        pass

class BasicMonitor(PretrainingMonitor):

    def __init__(self, out_dir : str, identifier : str):
        super().__init__()
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.identifier = identifier
        self.loss_train = []
        self.loss_valid = []
        self.error_rate = []
        self.best_error = []
        self.info = []

    def getState(self):
        return self.best_state

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, val):
        self._identifier = val

        self.net_pickle_path = os.path.join(self.out_dir, 'net_{0}.pkg'.format(self.identifier))
        # if we find a pickled net state with the same identifier we open it
        self.best_state = None
        if os.path.exists(self.net_pickle_path):
            self.best_state = torch.load(self.net_pickle_path)

        self.net_log_path = os.path.join(self.out_dir, 'net_{0}.log'.format(self.identifier))
        if hasattr(self, "log"):
            self.log.close()
        self.log = open(self.net_log_path, 'w')

    def finalizeStep(self, current_epoch: int, current_step: int, total_epochs: int, total_steps: int):
        self.info.append("Epoch: %d step: %d error_rate: %.3f loss_train: %.3f loss_valid %.3f" % (current_epoch, current_step, self.error_rate[-1], self.loss_train[-1], self.loss_valid[-1] if self.loss_valid[-1] is not None else np.inf))
        print(self.info[-1], file=self.log)
        self.log.flush()

    def performance(self, loss_train, loss_valid, error_rate, best_error):
        self.loss_train.append(loss_train)
        self.loss_valid.append(loss_valid)
        self.error_rate.append(error_rate)
        self.best_error.append(best_error)

    def smiles(self, smiles, is_valid):
        print('%d\t%s' % (is_valid, smiles), file=self.log)

    def model(self, model):
        pass

    def state(self, current_state, is_best=False):
        if is_best:
            self.best_state = current_state
            torch.save(self.best_state, self.net_pickle_path)

    def close(self):
        self.log.close()
        if self.best_state:
            torch.save(self.best_state, self.net_pickle_path)