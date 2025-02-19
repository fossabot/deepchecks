"""Module containing all the base classes for checks."""
import abc
import enum
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, List, Union, Dict, cast

__all__ = ['CheckResult', 'BaseCheck', 'SingleDatasetBaseCheck', 'CompareDatasetsBaseCheck', 'TrainTestBaseCheck',
           'ModelOnlyBaseCheck', 'ConditionResult', 'ConditionCategory', 'CheckFailure']

import pandas as pd
from IPython.core.display import display_html
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler

from deepchecks.base.display_pandas import display_dataframe
from deepchecks.utils.strings import split_camel_case
from deepchecks.errors import DeepchecksValueError


class Condition:
    """Contain condition attributes."""

    name: str
    function: Callable
    params: Dict

    def __init__(self, name: str, function: Callable, params: Dict):
        if not isinstance(function, Callable):
            raise DeepchecksValueError(f'Condition must be a function `(Any) -> Union[ConditionResult, bool]`, '
                                     f'but got: {type(function).__name__}')
        if not isinstance(name, str):
            raise DeepchecksValueError(f'Condition name must be of type str but got: {type(name).__name__}')
        self.name = name
        self.function = function
        self.params = params

    def __call__(self, *args, **kwargs) -> 'ConditionResult':
        result = cast(ConditionResult, self.function(*args, **kwargs))
        result.set_name(self.name)
        return result


class ConditionCategory(enum.Enum):
    """Condition result category. indicates whether the result should fail the suite."""

    FAIL = 'FAIL'
    WARN = 'WARN'


class ConditionResult:
    """Contain result of a condition function."""

    is_pass: bool
    category: ConditionCategory
    details: str
    name: str

    def __init__(self, is_pass: bool, details: str = '',
                 category: ConditionCategory = ConditionCategory.FAIL):
        """Initialize condition result.

        Args:
            is_pass (bool): Whether the condition functions passed the given value or not.
            details (str): What actually happened in the condition.
            category (ConditionCategory): The category to which the condition result belongs.
        """
        self.is_pass = is_pass
        self.details = details
        self.category = category

    def set_name(self, name: str):
        """Set name to be displayed in table.

        Args:
            name (str): Description of the condition to be displayed.
        """
        self.name = name

    def get_sort_value(self):
        """Return sort value of the result."""
        if self.is_pass:
            return 3
        elif self.category == ConditionCategory.FAIL:
            return 1
        else:
            return 2

    def get_icon(self):
        """Return icon of the result to display."""
        if self.is_pass:
            return '<div style="color: green;text-align: center">\U00002713</div>'
        elif self.category == ConditionCategory.FAIL:
            return '<div style="color: red;text-align: center">\U00002716</div>'
        else:
            return '<div style="color: orange;text-align: center;font-weight:bold">\U00000021</div>'

    def __repr__(self):
        """Return string representation for printing."""
        return str(vars(self))


class CheckResult:
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Attributes:
        value (Any): Value calculated by check. Can be used to decide if decidable check passed.
        display (Dict): Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    """

    value: Any
    header: str
    display: List[Union[Callable, str, pd.DataFrame, Styler]]
    condition_results: List[ConditionResult]

    def __init__(self, value, header: str = None, check=None, display: Any = None):
        """Init check result.

        Args:
            value (Any): Value calculated by check. Can be used to decide if decidable check passed.
            header (str): Header to be displayed in python notebook.
            check (Class): The check class which created this result. Used to extract the summary to be
            displayed in notebook.
            display (List): Objects to be displayed (dataframe or function or html)
        """
        self.value = value
        self.header = header or (check and check.name()) or None
        self.check = check
        self.condition_results = []

        if display is not None and not isinstance(display, List):
            self.display = [display]
        else:
            self.display = display or []

        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, Callable, Styler)):
                raise DeepchecksValueError(f'Can\'t display item of type: {type(item)}')

    def _ipython_display_(self):
        if self.header:
            display_html(f'<h4>{self.header}</h4>', raw=True)
        if self.check and '__doc__' in dir(self.check):
            docs = self.check.__doc__
            # Take first non-whitespace line.
            summary = next((s for s in docs.split('\n') if not re.match('^\\s*$', s)), '')
            display_html(f'<p>{summary}</p>', raw=True)

        for item in self.display:
            if isinstance(item, (pd.DataFrame, Styler)):
                display_dataframe(item)
            elif isinstance(item, str):
                display_html(item, raw=True)
            elif isinstance(item, Callable):
                item()
                plt.show()
            else:
                raise Exception(f'Unable to display item of type: {type(item)}')
        if not self.display:
            display_html('<p><b>&#x2713;</b> Nothing found</p>', raw=True)

    def __repr__(self):
        """Return default __repr__ function uses value."""
        return self.value.__repr__()

    def set_condition_results(self, results: List[ConditionResult]):
        """Set the conditions results for current check result."""
        self.conditions_results = results

    def have_conditions(self):
        """Return if this check have condition results."""
        return bool(self.conditions_results)

    def have_display(self):
        """Return if this check have dsiplay."""
        return bool(self.display)

    def passed_conditions(self):
        """Return if this check have not passing condition results."""
        return all((r.is_pass for r in self.conditions_results))

    def get_conditions_sort_value(self):
        """Get largest sort value of the conditions results."""
        return max([r.get_sort_value() for r in self.conditions_results])


class BaseCheck(metaclass=abc.ABCMeta):
    """Base class for check."""

    _conditions: OrderedDict
    _conditions_index: int

    def __init__(self):
        self._conditions = OrderedDict()
        self._conditions_index = 0

    def conditions_decision(self, result: CheckResult) -> List[ConditionResult]:
        """Run conditions on given result."""
        results = []
        condition: Condition
        for condition in self._conditions.values():
            output = condition.function(result.value, **condition.params)
            if isinstance(output, bool):
                output = ConditionResult(output)
            elif not isinstance(output, ConditionResult):
                raise DeepchecksValueError(f'Invalid return type from condition {condition.name}, got: {type(output)}')
            output.set_name(condition.name)
            results.append(output)
        return results

    def add_condition(self, name: str, condition_func: Callable[[Any], Union[ConditionResult, bool]], **params):
        """Add new condition function to the check.

        Args:
            name (str): Name of the condition. should explain the condition action and parameters
            condition_func (Callable[[Any], Union[List[ConditionResult], bool]]): Function which gets the value of the
                check and returns object of List[ConditionResult] or boolean.
            params: Additional parameters to pass when calling the condition function.
        """
        cond = Condition(name, condition_func, params)
        self._conditions[self._conditions_index] = cond
        self._conditions_index += 1
        return self

    def __repr__(self, tabs=0, prefix=''):
        """Representation of check as string.

        Args:
            tabs (int): number of tabs to shift by the output
        """
        tab_chr = '\t'
        params = self.params()
        if params:
            params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
            params_str = f'({params_str})'
        else:
            params_str = ''

        name = prefix + self.__class__.__name__
        check_str = f'{tab_chr * tabs}{name}{params_str}'
        if self._conditions:
            conditions_str = ''.join([f'\n{tab_chr * (tabs + 2)}{i}: {s.name}' for i, s in self._conditions.items()])
            return f'{check_str}\n{tab_chr * (tabs + 1)}Conditions:{conditions_str}'
        else:
            return check_str

    def params(self) -> Dict:
        """Return parameters to show when printing the check."""
        return {k: v for k, v in vars(self).items() if not k.startswith('_') and v is not None}

    def clean_conditions(self):
        """Remove all conditions from this check instance."""
        self._conditions.clear()
        self._conditions_index = 0

    def remove_condition(self, index: int):
        """Remove given condition by index.

        Args:
            index (int): index of condtion to remove
        """
        if index not in self._conditions:
            raise DeepchecksValueError(f'Index {index} of conditions does not exists')
        self._conditions.pop(index)

    @classmethod
    def name(cls):
        """Name of class in split camel case."""
        return split_camel_case(cls.__name__)


class SingleDatasetBaseCheck(BaseCheck):
    """Parent class for checks that only use one dataset."""

    @abc.abstractmethod
    def run(self, dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class CompareDatasetsBaseCheck(BaseCheck):
    """Parent class for checks that compare between two datasets."""

    @abc.abstractmethod
    def run(self, dataset, baseline_dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class TrainTestBaseCheck(BaseCheck):
    """Parent class for checks that compare two datasets.

    The class checks train dataset and test dataset for model training and test.
    """

    @abc.abstractmethod
    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Define run signature."""
        pass


class ModelOnlyBaseCheck(BaseCheck):
    """Parent class for checks that only use a model and no datasets."""

    @abc.abstractmethod
    def run(self, model) -> CheckResult:
        """Define run signature."""
        pass


@dataclass
class CheckFailure:
    """Class which holds a run exception of a check."""

    check: Any  # Check class, not instance
    exception: Exception
