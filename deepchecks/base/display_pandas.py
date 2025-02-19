"""Handle displays of pandas objects."""
from typing import Union
import warnings

from IPython.core.display import display_html
import pandas as pd
__all__ = ['display_dataframe', 'dataframe_to_html']

from pandas.io.formats.style import Styler


def display_dataframe(df: Union[pd.DataFrame, Styler]):
    """Display in IPython given dataframe.

    Args:
        df (Union[pd.DataFrame, Styler]): Dataframe to display
    """
    display_html(dataframe_to_html(df), raw=True)


def dataframe_to_html(df: Union[pd.DataFrame, Styler]):
    """Convert dataframe to html.

    Args:
        df (Union[pd.DataFrame, Styler]): Dataframe to convert to html
    """
    try:
        if isinstance(df, pd.DataFrame):
            df_styler = df.style
        else:
            df_styler = df
        # Using deprecated pandas method so hiding the warning
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df_styler.set_precision(2)

        # Align everything to the left
        df_styler.set_table_styles([dict(selector='table,thead,tbody,th,td', props=[('text-align', 'left')])])
        return df_styler.render()
    # Because of MLC-154. Dataframe with Multi-index or non unique indices does not have a style
    # attribute, hence we need to display as a regular pd html format.
    except ValueError:
        return df.to_html()
