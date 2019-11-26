#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is mainly for environment (predictor) construction.

It contains machine learning (ML) training, cross validation and independent set test.
The traditional ML models are implemented by using Scikit-Learn (version >= 0.18)
"""

import os

import click
import numpy as np
import pandas as pd
import torch as T
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader, TensorDataset

from drugex.core import util
from drugex.api.environ.data import EnvironData
from drugex.api.environ.models import RF
from drugex.api.environ.serialization import FileEnvSerializer
from drugex.core.model import MTFullyConnected, STFullyConnected


def DNN(X, y, X_ind, y_ind, out, is_regression=False, *, batch_size, n_epoch, lr):
    """Cross Validation and independent set test for fully connected deep neural network

    Arguments:
        X (ndarray): Feature data of training and validation set for cross-validation.
                     m X n matrix, m is the No. of samples, n is the No. of fetures
        y (ndarray): Label data of training and validation set for cross-validation.
                     m X t matrix if it is for multi-task model,
                     m is the No. of samples, n is the No. of tasks or classes;
                     m-D vector if it is only for single task model, and m is the No. of samples.
        X_ind (ndarray): Feature data of independent test set for independent test.
                         It has the similar data structure as X.
        y_ind (ndarray): Feature data of independent set for for independent test.
                         It has the similar data structure as y
        out (str): The file path for saving the result data.
        is_regression (bool, optional): define the model for regression (True) or classification (False) (Default: False)

    Returns:
         cvs (ndarray): cross-validation results. If it is single task, the shape is (m, ),
                        m is the No. of samples, it contains real label and probability value;
                        if it is multi-task, the shape is m X n, n is the No. of tasks.
         inds (ndarray): independent test results. It has similar data structure as cvs.
    """
    if 'mtqsar' in out or is_regression:
        folds = KFold(5).split(X)
        NET = MTFullyConnected
    else:
        folds = StratifiedKFold(5).split(X, y[:, 0])
        NET = STFullyConnected
    indep_set = TensorDataset(T.Tensor(X_ind), T.Tensor(y_ind))
    indep_loader = DataLoader(indep_set, batch_size=batch_size)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        train_set = TensorDataset(T.Tensor(X[trained]), T.Tensor(y[trained]))
        train_loader = DataLoader(train_set, batch_size=batch_size)
        valid_set = TensorDataset(T.Tensor(X[valided]), T.Tensor(y[valided]))
        valid_loader = DataLoader(valid_set, batch_size=batch_size)
        net = NET(X.shape[1], y.shape[1], is_reg=is_regression)
        net.fit(train_loader, valid_loader, out='%s_%d' % (out, i), epochs=n_epoch, lr=lr)
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)
    cv, ind = y == y, y_ind == y_ind
    return cvs[cv], inds[ind] / 5

def SVM(X, y, X_ind, y_ind, is_reg=False):
    """Cross Validation and independent set test for Support Vector Machine (SVM)

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
        is_reg (bool, optional): define the model for regression (True) or classification (False) (Default: False)

    Returns:
         cvs (ndarray): cross-validation results. The shape is (m, ), m is the No. of samples.
         inds (ndarray): independent test results. It has similar data structure as cvs.
    """
    if is_reg:
        folds = KFold(5).split(X)
        model = SVR()
    else:
        folds = StratifiedKFold(5).split(X, y)
        model = SVC(probability=True)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    gs = GridSearchCV(model, {'C': 2.0 ** np.array([-5, 15]), 'gamma': 2.0 ** np.array([-15, 5])}, n_jobs=5)
    gs.fit(X, y)
    params = gs.best_params_
    print(params)
    for i, (trained, valided) in enumerate(folds):
        model = SVC(probability=True, C=params['C'], gamma=params['gamma'])
        model.fit(X[trained], y[trained])
        if is_reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def KNN(X, y, X_ind, y_ind, is_reg=False):
    """Cross Validation and independent set test for KNN.

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
        is_reg (bool, optional): define the model for regression (True) or classification (False) (Default: False)

    Returns:
         cvs (ndarray): cross-validation results. The shape is (m, ), m is the No. of samples.
         inds (ndarray): independent test results. It has similar data structure as cvs.
    """
    if is_reg:
        folds = KFold(5).split(X)
        alg = KNeighborsRegressor
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = KNeighborsClassifier
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = alg(n_jobs=1)
        model.fit(X[trained], y[trained])
        if is_reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def NB(X, y, X_ind, y_ind):
    """Cross Validation and independent set test for Naive Bayes.

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

    Returns:
         cvs (ndarray): cross-validation results. The shape is (m, ), m is the No. of samples.
         inds (ndarray): independent test results. It has similar data structure as cvs.
    """
    folds = StratifiedKFold(5).split(X, y)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = GaussianNB()
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


class ChEMBLCSV(EnvironData):

    def __init__(self, input_file : str, activity_threshold=None, subsample_size=None):
        super().__init__(subsample_size=subsample_size)
        self.PAIR = ['CMPD_CHEMBLID', 'CANONICAL_SMILES', 'PCHEMBL_VALUE', 'ACTIVITY_COMMENT']
        self.activity_threshold = activity_threshold
        if self.activity_threshold is not None:
            self.is_regression = False
        else:
            self.is_regression = True
        self.input_file = input_file
        self.df = pd.DataFrame()
        self.X = None
        self.y = None
        self.update()

    def getSMILES(self):
        return self.df.CANONICAL_SMILES

    def update(self):
        if self.subsample_size is not None:
            self.df = pd.read_table(self.input_file).sample(self.subsample_size)
        else:
            self.df = pd.read_table(self.input_file)
        self.df = self.df[self.PAIR].set_index(self.PAIR[0])
        self.df[self.PAIR[2]] = self.df.groupby(self.df.index).mean()

        # The molecules that have PChEMBL value
        numery = self.df[self.PAIR[1:-1]].drop_duplicates().dropna()
        if self.is_regression:
            self.df = numery
            self.y = numery[self.PAIR[2:3]].values
        else:
            # The molecules that do not have PChEMBL value
            # but has activity comment to show whether it is active or not.
            binary = self.df[self.df.ACTIVITY_COMMENT.str.contains('Active') == True].drop_duplicates()
            binary = binary[~binary.index.isin(numery.index)]
            # binary.loc[binary.ACTIVITY_COMMENT == 'Active', 'PCHEMBL_VALUE'] = 100.0
            binary.loc[binary.ACTIVITY_COMMENT.str.contains('Not'), 'PCHEMBL_VALUE'] = 0.0
            binary = binary[self.PAIR[1:3]].dropna().drop_duplicates()
            self.df = numery.append(binary)
            # For is_regression model the active ligand is defined as
            # PChBMBL value >= activity_threshold
            self.y = (self.df[self.PAIR[2:3]] >= self.activity_threshold).astype(float).values

        # ECFP6 fingerprints extraction
        self.X = util.Environment.ECFP_from_SMILES(self.df.CANONICAL_SMILES).values
    def getX(self):
        return self.X

    def gety(self):
        return self.y

    def getGroundTruthData(self):
        data = pd.DataFrame()
        data['CANONICAL_SMILES'], data['LABEL'] = self.getSMILES(), self.gety()[:, 0]
        return data


# Model performance and saving
def _main_helper(*, path, feat, alg, is_regression, batch_size, n_epoch, lr, output):
    os.makedirs(output, exist_ok=True)
    out = os.path.join(output, )

    # Model training and saving with RF
    # TODO: unify all algorithms under this interface
    data_chembl = ChEMBLCSV(path, 6.5)
    if alg == "RF":
        model = RF(train_provider=data_chembl, test_provider=data_chembl)
        model.fit()
        ser = FileEnvSerializer(out_dir=output, identifier='%s_%s_%s' % (alg, 'reg' if is_regression else 'cls', feat), include_perf=True)
        model.save(serializer=ser)
        return

    X = data_chembl.getX()
    y = data_chembl.gety()
    # Cross validation and independent test
    data = data_chembl.getGroundTruthData()
    test = data_chembl.getGroundTruthData()
    if alg == 'SVM':
        cv, ind = SVM(X, y[:, 0], X, y[:, 0])
    elif alg == 'KNN':
        cv, ind = KNN(X, y[:, 0], X, y[:, 0])
    elif alg == 'NB':
        cv, ind = NB(X, y[:, 0], X, y[:, 0])
    elif alg == 'DNN':
        cv, ind = DNN(X, y, X, y, out=out, is_regression=is_regression, batch_size=batch_size, n_epoch=n_epoch, lr=lr)
    else:
        raise ValueError('Invalid algorithm: {}'.format(alg))

    data['SCORE'], test['SCORE'] = cv, ind
    data.to_csv(out + '.cv.txt', index=None)
    test.to_csv(out + '.ind.txt', index=None)


@click.command()
@click.option('-p', '--path', type=click.Path(file_okay=True), default='data/FT_ENV_data.txt')
@click.option('-o', '--output', type=click.Path(file_okay=False, dir_okay=True), default='output')
@click.option('--lr', type=float, default=1e-5, show_default=True)
@click.option('--batch-size', type=int, default=1024, show_default=True)
@click.option('--n-epoch', type=int, default=1000, show_default=True)
@click.option('--n-threads', type=int, default=1, show_default=True)
@click.option('--regression', is_flag=True)
@click.option('-a', '--algorithm', type=click.Choice(['SVM', 'KNN', 'RF', 'DNN', 'NB']), default='RF', show_default=True)
@click.option('--cuda', is_flag=True)
def main(path, output, lr, batch_size, n_epoch, n_threads, regression, algorithm, cuda: bool):
    T.set_num_threads(n_threads)
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _main_helper(
        path=path,
        feat='ecfp6',
        alg=algorithm,
        is_regression=regression,
        batch_size=batch_size,
        n_epoch=n_epoch,
        lr=lr,
        output=output,
    )


if __name__ == '__main__':
    main()
