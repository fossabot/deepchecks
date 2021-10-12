from re import I

import numpy as np

from mlchecks.checks.overview.performance.performance_report import PreformaceReport
from hamcrest import *

from mlchecks.utils import MLChecksValueError


def assert_model_result(result):
    res_val = result.value
    confusion_matrix = res_val.pop('confusion_matrix')
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            assert(isinstance(confusion_matrix[i][j] , np.int64))
    macro_performance = res_val.pop('macro_performance')
    for col in macro_performance.values():
        for val in col.values():
            assert(isinstance(val , float))
    for value in res_val.values():
        assert(isinstance(value , np.float64))

def test_model_info_object(iris_dataset):
    (ds, model) = iris_dataset

    # Arrange
    check = PreformaceReport()
    # Act X
    result = check.run(ds, model) 
    # Assert
    assert_model_result(result)
