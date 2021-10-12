import base64
import io
from typing import Any
from matplotlib import pyplot as plt
import sklearn
import catboost

__all__ = ['SUPPORTED_BASE_MODELS', 'MLChecksValueError', 'model_type_validation']


SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]


class MLChecksValueError(ValueError):
    pass


def model_type_validation(model: Any):
    """Receive any object and check if it's an instance of a model we support

    Raises
        MLChecksException: If the object is not of a supported type
    """
    if not any([isinstance(model, base) for base in SUPPORTED_BASE_MODELS]):
        raise MLChecksValueError(f'Model must inherit from one of supported models: {SUPPORTED_BASE_MODELS}')


def get_plt_base64():
    plt_buffer = io.BytesIO()
    plt.savefig(plt_buffer, format='jpg')
    plt_buffer.seek(0)
    return base64.b64encode(plt_buffer.read()).decode("utf-8")
