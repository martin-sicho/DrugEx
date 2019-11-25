"""
models

Created by: Martin Sicho
On: 19-11-19, 9:59
"""
import numpy as np

from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from drugex.api.environ.data import EnvironData
from drugex.api.environ.serialization import EnvironSerializer, EnvironDeserializer
from drugex.core import util

class Environ(ABC):

    @staticmethod
    def load(deserializer : EnvironDeserializer):
        return deserializer.getModel()

    def __init__(self, train_provider : EnvironData, test_provider=None):
        self.train_provider = train_provider
        self.test_provider = test_provider

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predictSMILES(self, smiles):
        pass

    @abstractmethod
    def getPerfData(self):
        pass

    def save(self, serializer : EnvironSerializer):
        serializer.saveModel(self)


class RF(Environ):
    """Cross Validation and independent set test for Random Forest model

    Arguments:
        X (ndarray): Feature data of training and validation set for cross-validation.
                     m X n matrix, m is the No. of samples, n is the No. of fetures
        y (ndarray): Label data of training and validation set for cross-validation.
                     m-D vector, and m is the No. of samples.
        X_ind (ndarray): Feature data of independent test set for independent test.
                         It has the similar data structure as X.
        y_ind (ndarray): Feature data of independent set for for independent test.
                         It has the similar data structure as y
        out (str): The file path for saving the result data.
        is_regression (bool, optional): define the model for regression (True) or classification (False) (Default: False)

    Returns:
         cvs (ndarray): cross-validation results. The shape is (m, ), m is the No. of samples.
         inds (ndarray): independent test results. It has similar data structure as cvs.
        """

    def __init__(self, train_provider : EnvironData, test_provider=None, n_folds=5, params={"n_estimators" : 1000, "n_jobs" : 10}):
        super().__init__(train_provider, test_provider)
        if not self.test_provider:
            self.test_provider = train_provider
        self.is_regression = train_provider.is_regression
        self.n_folds = n_folds
        self.params = params
        self.cvs = None
        self.inds = None
        self.alg = RandomForestRegressor if self.is_regression else RandomForestClassifier
        self.model = None

    def fit(self):
        X = self.train_provider.getX()
        y = self.train_provider.gety()[:, 0]
        X_ind = self.test_provider.getX()
        y_ind = self.test_provider.gety()[:, 0]

        if self.is_regression:
            folds = KFold(self.n_folds).split(X)
        else:
            folds = StratifiedKFold(self.n_folds).split(X, y)
        self.cvs = np.zeros(y.shape)
        self.inds = np.zeros(y_ind.shape)
        for i, (trained, valided) in enumerate(folds):
            model = self.alg(**self.params)
            model.fit(X[trained], y[trained])
            if self.is_regression:
                self.cvs[valided] = model.predict(X[valided])
                self.inds += model.predict(X_ind)
            else:
                self.cvs[valided] = model.predict_proba(X[valided])[:, 1]
                self.inds += model.predict_proba(X_ind)[:, 1]
        self.inds = self.inds / self.n_folds

        # fit the model on the whole set
        self.model = self.alg(**self.params)
        self.model.fit(X, y)

        return self.getPerfData()

    def predictSMILES(self, smiles):
        fps = util.Environment.ECFP_from_SMILES(smiles)
        if self.is_regression:
            preds = self.model.predict(fps)
        else:
            preds = self.model.predict_proba(fps)[:, 1]
        return preds

    def getPerfData(self):
        cv = self.train_provider.getGroundTruthData()
        test = self.test_provider.getGroundTruthData()
        cv['SCORE'], test['SCORE'] = self.cvs, self.inds
        return cv, test