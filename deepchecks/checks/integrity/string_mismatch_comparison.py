"""String mismatch functions."""
from collections import defaultdict
from typing import Union, Iterable

import pandas as pd

from deepchecks import CheckResult, Dataset, ensure_dataframe_type, CompareDatasetsBaseCheck, ConditionResult
from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.strings import get_base_form_to_variants_dict, is_string_column, format_percent, \
    format_columns_for_condition
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_df


__all__ = ['StringMismatchComparison']


def _condition_percent_limit(result, ratio: float):
    not_passing_columns = {}
    for col, baseforms in result.items():
        sum_percent = 0
        for info in baseforms.values():
            sum_percent += info['percent_variants_only_in_tested']
        if sum_percent > ratio:
            not_passing_columns[col] = format_percent(sum_percent)

    if not_passing_columns:
        details = f'Found columns with variants over ratio: {not_passing_columns}'
        return ConditionResult(False, details)
    return ConditionResult(True)


def percentage_in_series(series, values):
    count = sum(series.isin(values))
    percent = count / series.size
    return percent, f'{format_percent(percent)} ({count})'


class StringMismatchComparison(CompareDatasetsBaseCheck):
    """Detect different variants of string categories between the same categorical column in two datasets.

    This check compares the same categorical column within a dataset and baseline and checks whether there are
    variants of similar strings that exists only in dataset and not in baseline.
    Specifically, we define similarity between strings if they are equal when ignoring case and non-letter
    characters.
    Example:
    We have a baseline dataset with similar strings 'string' and 'St. Ring', which have different meanings.
    Our tested dataset has the strings 'string', 'St. Ring' and a new phrase, 'st.  ring'.
    Here, we have a new variant of the above strings, and would like to be acknowledged, as this is obviously a
    different version of 'St. Ring'.

     Args:
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
          ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
          variable
        n_top_columns (int): (optional - used only if model was specified)
          amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 n_top_columns: int = 10):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run(self, dataset, baseline_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): A dataset object.
            baseline_dataset (Dataset): A dataset object.
            model: Not used in this check.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._string_mismatch_comparison(dataset, baseline_dataset, feature_importances)

    def _string_mismatch_comparison(self, dataset: Union[pd.DataFrame, Dataset],
                                   baseline_dataset: Union[pd.DataFrame, Dataset],
                                   feature_importances: pd.Series=None) -> CheckResult:
        # Validate parameters
        df: pd.DataFrame = ensure_dataframe_type(dataset)
        df = filter_columns_with_validation(df, self.columns, self.ignore_columns)
        baseline_df: pd.DataFrame = ensure_dataframe_type(baseline_dataset)

        display_mismatches = []
        result_dict = defaultdict(dict)

        # Get shared columns
        columns = set(df.columns).intersection(baseline_df.columns)

        for column_name in columns:
            tested_column: pd.Series = df[column_name]
            baseline_column: pd.Series = baseline_df[column_name]
            # If one of the columns isn't string type, continue
            if not is_string_column(tested_column) or not is_string_column(baseline_column):
                continue

            tested_baseforms = get_base_form_to_variants_dict(tested_column.unique())
            baseline_baseforms = get_base_form_to_variants_dict(baseline_column.unique())

            common_baseforms = set(tested_baseforms.keys()).intersection(baseline_baseforms.keys())
            for baseform in common_baseforms:
                tested_values = tested_baseforms[baseform]
                baseline_values = baseline_baseforms[baseform]
                # If at least one unique value in tested dataset, add the column to results
                if len(tested_values - baseline_values) > 0:
                    # Calculate all values to be shown
                    variants_only_in_dataset = list(tested_values - baseline_values)
                    variants_only_in_baseline = list(baseline_values - tested_values)
                    common_variants = list(tested_values & baseline_values)
                    percent_variants_only_in_dataset = percentage_in_series(tested_column, variants_only_in_dataset)
                    percent_variants_in_baseline = percentage_in_series(baseline_column, variants_only_in_baseline)

                    display_mismatches.append([column_name, baseform, common_variants,
                                               variants_only_in_dataset, percent_variants_only_in_dataset[1],
                                               variants_only_in_baseline, percent_variants_in_baseline[1]])
                    result_dict[column_name][baseform] = {
                        'commons': common_variants, 'variants_only_in_tested': variants_only_in_dataset,
                        'variants_only_in_baseline': variants_only_in_baseline,
                        'percent_variants_only_in_tested': percent_variants_only_in_dataset[0],
                        'percent_variants_in_baseline': percent_variants_in_baseline[0]
                    }

        # Create result dataframe
        if display_mismatches:
            df_graph = pd.DataFrame(display_mismatches,
                                    columns=['Column name', 'Base form', 'Common variants', 'Variants only in dataset',
                                             '% Unique variants out of all dataset samples (count)',
                                             'Variants only in baseline',
                                             '% Unique variants out of all baseline samples (count)'])
            df_graph = df_graph.set_index(['Column name', 'Base form'])
            df_graph = column_importance_sorter_df(df_graph, dataset, feature_importances,
                                                   self.n_top_columns, col='Column name')
            # For display transpose the dataframe
            display = df_graph.T
        else:
            display = None

        return CheckResult(result_dict, check=self.__class__, display=display)

    def add_condition_no_new_variants(self):
        """Add condition - no new variants allowed in test data."""
        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        name = f'No new variants allowed in test data for {column_names}'
        return self.add_condition(name, _condition_percent_limit, ratio=0)

    def add_condition_ratio_new_variants_not_more_than(self, ratio: float):
        """Add condition - no new variants allowed above given percentage in test data.

        Args:
            ratio (float): Max percentage of new variants in test data allowed.
        """
        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        name = f'Not more than {format_percent(ratio)} new variants in test data for {column_names}'
        return self.add_condition(name, _condition_percent_limit, ratio=ratio)
