"""
generator

Created by: Martin Sicho
On: 25-11-19, 15:13
"""
from abc import ABC, abstractmethod

import pandas as pd

from drugex.api.environ.models import Environ
from drugex.api.pretrain.generators import Generator


class Consumer(ABC):

    @abstractmethod
    def __call__(self, smiles, is_valid, score):
        pass

class CSVConsumer(Consumer):

    def __init__(self, out_path):
        self.out = out_path
        self.df = pd.DataFrame(columns=['CANONICAL_SMILES', 'IS_VALID','SCORE'])

    def __call__(self, smiles, is_valid, score):
        self.df = self.df.append(
            pd.DataFrame({
                'CANONICAL_SMILES' : [smiles]
                , 'SCORE' : [score]
                , 'IS_VALID' : [bool(is_valid)]
            })
        )

    def save(self):
        self.df.to_csv(self.out, sep='\t', index=None)

class Designer(ABC):

    def __init__(self, agent : Generator, consumer : Consumer, environ : Environ, batch_size : int, n_samples : int):
        self.agent = agent
        self.consumer = consumer
        self.environ = environ
        self.batch_size = batch_size
        self.n_samples = n_samples

    def __call__(self):
        pass

class BasicDesigner(Designer):

    def __init__(self, agent: Generator, consumer: Consumer, environ=None, batch_size=512, n_samples=10000):
        super().__init__(agent, consumer, environ, batch_size, n_samples)

    def __call__(self):
        """Generate novel molecules with SMILES representation and store them into hard drive as a data frame.

        Arguments:
            agent_path (str): the neural states file paths for the RNN agent (generator).
            out (str): file path for the generated molecules (and scores given by environment).
            num (int, optional): the total No. of SMILES that need to be generated. (Default: 10000)
            environment_path (str, optional): the file path of the predictor for environment construction.
        """

        for i in range(self.n_samples // self.batch_size + 1):
            if i == 0 and self.n_samples % self.batch_size == 0: continue
            smiles, valids = self.agent.sample(self.batch_size if i != 0 else self.n_samples % self.batch_size)
            scores = []
            if self.environ:
                # calculating the reward of each SMILES based on the environment (predictor).
                scores = self.environ.predictSMILES(smiles)
            else:
                scores = [None] * len(smiles)

            for smi, valid, score in zip(smiles, valids, scores):
                self.consumer(smi, valid, score)
