"""Utils module containing utilities for checks working with metrics."""
import enum
from numbers import Number
from typing import List, Union, Dict, Callable

import numpy as np
from sklearn.metrics import get_scorer, make_scorer, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.base import ClassifierMixin, RegressorMixin

__all__ = ['ModelType', 'task_type_check', 'get_metrics_list', 'validate_scorer', 'DEFAULT_METRICS_DICT',
           'DEFAULT_SINGLE_METRIC', 'get_metrics_ratio']

from deepchecks.utils import model_type_validation, DeepchecksValueError


class ModelType(enum.Enum):
    """Enum containing suppoerted task types."""

    REGRESSION = 'regression'  # regression
    BINARY = 'binary'  # binary classification
    MULTICLASS = 'multiclass'  # multiclass classification


DEFAULT_BINARY_METRICS = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0)
}

DEFAULT_MULTICLASS_METRICS = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision - Macro Average': make_scorer(precision_score, average='macro', zero_division=0),
    'Recall - Macro Average': make_scorer(recall_score, average='macro', zero_division=0)
}

DEFAULT_REGRESSION_METRICS = {
    'RMSE': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
}

DEFAULT_SINGLE_METRIC = {
    ModelType.BINARY: 'Accuracy',
    ModelType.MULTICLASS: 'Accuracy',
    ModelType.REGRESSION: 'RMSE'
}

DEFAULT_METRICS_DICT = {
    ModelType.BINARY: DEFAULT_BINARY_METRICS,
    ModelType.MULTICLASS: DEFAULT_MULTICLASS_METRICS,
    ModelType.REGRESSION: DEFAULT_REGRESSION_METRICS
}


def task_type_check(model: Union[ClassifierMixin, RegressorMixin], dataset: 'Dataset') -> ModelType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:
        TaskType enum corresponding to the model and dataset
    """
    model_type_validation(model)
    dataset.validate_label(task_type_check.__name__)

    if hasattr(model, 'predict_proba'):
        unique_labels = dataset.label_col().unique()
        if sorted(unique_labels) != list(range(len(unique_labels))):
            raise DeepchecksValueError(f'Classification labels must be a consecutive set from 0 to MAX_LABEL,'
                                       f' found {sorted(unique_labels)}.')
        if dataset.label_col().nunique() > 2:
            return ModelType.MULTICLASS
        else:
            return ModelType.BINARY
    else:
        return ModelType.REGRESSION


def task_type_validation(model: Union[ClassifierMixin, RegressorMixin], dataset: 'Dataset',
                         expected_types: List[ModelType], check_name: str = None):
    """Validate task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels
        expected_types (List[ModelType]): allowed types of model
        check_name (str): check name to print in error

    Raises:
            DeepchecksValueError if model type doesn't match one of the expected_types
    """
    task_type = task_type_check(model, dataset)
    if not task_type in expected_types:
        if check_name:
            prefix = f'Check {check_name} '
        else:
            prefix = ''
        raise DeepchecksValueError(f'{prefix}Expected model to be a type from {[e.value for e in expected_types]},'
                                 f' but received model of type: {task_type.value}')


def get_metrics_list(model, dataset: 'Dataset', alternative_metrics: Dict[str, Callable] = None
                     ) -> Dict[str, Callable]:
    """Return list of scorer objects to use in a metrics-dependant check.

    If no alternative_metrics is supplied, then a default list of metrics is used per task type, as it is inferred
    from the dataset and model. If a list is supplied, then the scorer functions are checked and used instead.

    Args:
        model (BaseEstimator): Model object for which the metrics would be calculated
        dataset (Dataset): Dataset object on which the metrics would be calculated
        alternative_metrics (Dict[str, Callable]): Optional dictionary of sklearn scorers to use instead of default list

    Returns:
        Dictionary containing names of metrics and scorer functions for the metrics.
    """
    if alternative_metrics:
        metrics = {}
        for name, scorer in alternative_metrics.items():
            metrics[name] = validate_scorer(scorer, model, dataset)
    else:
        # Check for model type
        model_type = task_type_check(model, dataset)
        metrics = DEFAULT_METRICS_DICT[model_type]

    return metrics


def validate_scorer(scorer, model, dataset):
    """If string, get scorer from sklearn. If callable, do heuristic to see if valid."""
    # Borrowed code from:
    # https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/metrics/_scorer.py#L421
    if isinstance(scorer, str):
        return get_scorer(scorer)
    elif callable(scorer):
        # Check that scorer runs for given model and data
        assert isinstance(scorer(model, dataset.data[dataset.features()].head(2), dataset.label_col().head(2)),
                          Number)
        return scorer


def get_metrics_ratio(train_metric: float, test_metric: float, max_ratio=np.Inf):
    """Return the ratio of test metric compared to train metric."""
    if train_metric == 0:
        return max_ratio
    else:
        ratio = test_metric / train_metric
        if train_metric < 0 and test_metric < 0:
            ratio = 1 / ratio
        ratio = min(max_ratio, ratio)
        return ratio
