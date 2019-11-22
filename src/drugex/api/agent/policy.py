"""
policy

Created by: Martin Sicho
On: 22-11-19, 13:30
"""
from abc import ABC, abstractmethod

from drugex.api.environ.models import Environ
from drugex.api.pretrain.generators import Generator


class PolicyGradient(ABC):

    def __init__(self, batch_size=512, mc=10, epsilon=0.01, beta=0.1):
        self.batch_size = batch_size
        self.mc = mc
        self.epsilon = epsilon
        self.beta = beta

    @abstractmethod
    def __call__(self, environ : Environ, exploit : Generator, explore = None):
        pass
