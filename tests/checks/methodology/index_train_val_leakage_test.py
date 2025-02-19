"""
Contains unit tests for the single_feature_contribution check
"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, calling, raises, has_items

from deepchecks import Dataset
from deepchecks.checks.methodology.index_leakage import IndexTrainTestLeakage
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def dataset_from_dict(d: dict, index_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, index=index_name)


def test_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.25, 0.01))


def test_limit_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 3, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage(n_index_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.5, 0.01))


def test_no_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [20, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0, 0.01))


def test_dataset_wrong_input():
    x = 'wrong_input'
    assert_that(
        calling(IndexTrainTestLeakage().run).with_args(x, x),
        raises(DeepchecksValueError, 'Check IndexTrainTestLeakage requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(IndexTrainTestLeakage().run).with_args(ds, ds),
        raises(DeepchecksValueError, 'Check IndexTrainTestLeakage requires dataset to have an index column'))


def test_nan():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11, np.nan]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7, np.nan]}, 'col1')
    check_obj = IndexTrainTestLeakage(n_index_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.2, 0.01))


def test_condition_leakage_fail():
    # Arrange
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11, np.nan]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7, np.nan]}, 'col1')
    check = IndexTrainTestLeakage(n_index_to_show=1).add_condition_ratio_not_greater_than(max_ratio=0.19)

    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 20.00% of index leakage',
                               name='Index leakage is not greater than 19.00%')
    ))


def test_condition_leakage_passesl():
    # Arrange
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [20, 5, 6, 7]}, 'col1')
    check = IndexTrainTestLeakage(n_index_to_show=1).add_condition_ratio_not_greater_than()

    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Index leakage is not greater than 0%')
    ))
