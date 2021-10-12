import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

from mlchecks.base import Dataset


@pytest.fixture(scope='session')
def iris():
    return load_iris(as_frame=True)


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    clf = AdaBoostClassifier()
    X = iris.data
    Y = iris.target
    clf.fit(X, Y)
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
