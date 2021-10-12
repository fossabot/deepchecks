"""Represents fixtures for unit testing using pytest."""
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
import pandas as pd
from mlchecks import Dataset

from mlchecks.base import Dataset


@pytest.fixture(scope='session')
def iris():
<<<<<<< HEAD
    return load_iris(as_frame=True)
=======
    df = load_iris(return_X_y=False, as_frame=True)
    return pd.concat([df.data, df.target], axis=1)


@pytest.fixture(scope='session')
def iris_dataset(iris):
    return Dataset(iris)
>>>>>>> a7557b92d8ab56d277d7b369b8150ba2025949c9


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    clf = AdaBoostClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf

@pytest.fixture(scope='session')
def iris_dataset(iris):
    clf = AdaBoostClassifier()
    frame = iris.frame
    X = iris.data
    Y = iris.target
    ds = Dataset(frame, 
                features=iris.feature_names,
                label='target')
    clf.fit(X, Y)
    return ds, clf
