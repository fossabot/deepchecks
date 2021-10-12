"""Represents fixtures for unit testing using pytest."""
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
import pandas as pd
from mlchecks import Dataset

from mlchecks.base import Dataset


@pytest.fixture(scope='session')
def iris():
    return load_iris(as_frame=True)


@pytest.fixture(scope='session')
def iris_dataset(iris):
    return Dataset(iris.frame)


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    clf = AdaBoostClassifier()
    features = iris.frame.drop('target', axis=1)
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
