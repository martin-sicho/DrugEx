"""
loggers

Created by: Martin Sicho
On: 11/24/19, 7:05 PM
"""
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError
from matplotlib import pyplot as plt

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
    def finalizeStep(self
         , current_epoch: int
         , current_batch : int
         , current_step: int
         , total_epochs: int
         , total_batches : int
         , total_steps: int
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
        self.loss_train = None
        self.loss_valid = None
        self.error_rate = None
        self.best_error = None
        self.step = None
        self.epoch = None
        self.info = None

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

        # open the log file
        self.net_log_path = os.path.join(self.out_dir, 'net_{0}.log'.format(self.identifier))
        if hasattr(self, "log"):
            self.log.close()
        self.log = open(self.net_log_path, 'a' if self.best_state else 'w')

        # open the CSV file to save training progress
        self.net_csv_path = os.path.join(self.out_dir, 'net_{0}.csv'.format(self.identifier))
        if hasattr(self, "csv"):
            self.csv.close()
        self.csv =  open(self.net_csv_path, 'a' if self.best_state else 'w')

        last = None
        try:
            self.csv_empty = False
            last = pd.read_table(self.net_csv_path, sep="\t", header=0)
            last = last.iloc[-1,:]
        except EmptyDataError:
            self.csv_empty = True
        if last is not None:
            self.last_epoch = last["EPOCH"]
            self.last_step = last["STEP"]
            self.last_run = last["RUN"]
        else:
            self.last_run = 0
            self.last_epoch = 0
            self.last_step = 0

    def finalizeStep(self
             , current_epoch: int
             , current_batch : int
             , current_step: int
             , total_epochs: int
             , total_batches : int
             , total_steps: int
        ):
        current_step = current_step + self.last_step
        current_epoch = current_epoch + self.last_epoch

        self.info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f loss_valid %.3f" % (current_epoch, current_step, self.error_rate, self.loss_train, self.loss_valid if self.loss_valid is not None else np.inf)
        print(self.info, file=self.log)
        self.log.flush()

        self.step = current_step
        self.epoch = current_epoch
        df = pd.DataFrame(
            {
                "STEP" : [current_step]
                , "EPOCH" : [current_epoch]
                , "LOSS_TRAIN" : [self.loss_train]
                , "LOSS_VALID" : [self.loss_valid]
                , "ERROR_RATE" : [self.error_rate]
                , "RUN" : self.last_run+1
            }
        )
        if self.csv_empty:
            df.to_csv(self.csv, header=True, index=False, sep="\t")
            self.csv_empty = False
        else:
            df.to_csv(self.csv, header=False, index=False, sep="\t")
        self.csv.flush()
        plt.close(self.getPerfFigure())

    def performance(self, loss_train, loss_valid, error_rate, best_error):
        self.loss_train = loss_train
        self.loss_valid = loss_valid
        self.error_rate = error_rate
        self.best_error = best_error

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
        self.csv.close()
        if self.best_state:
            torch.save(self.best_state, self.net_pickle_path)

    def getPerfFigure(self, save_png=True):
        try:
            df = pd.read_table(self.net_csv_path, sep="\t", header=0)
        except EmptyDataError:
            print("No data to generate the figure from...")
            return

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(df["STEP"], df["LOSS_TRAIN"] / 100, c='b', label='training loss')
        if self.loss_valid:
            ax1.plot(df["STEP"], df["LOSS_VALID"] / 100, c='r', label='validation loss')
        ax1.plot(df["STEP"], 1 - df["ERROR_RATE"], c='g', label='SMILES validity')
        plt.legend(loc='upper right')
        if save_png:
            plt.savefig(os.path.join(self.out_dir, "net_{0}.png".format(self.identifier)))
        return fig