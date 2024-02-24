"""
Library of functions for data analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame


def remove_outliers(df: DataFrame, column: str) -> DataFrame:
    """
    Remove outliers from a DataFrame  using the IQR method.

    Returns a new DataFrame with the outliers removed.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from
    column : str
        The column to remove outliers from

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the outliers removed
    """
    if not isinstance(df, DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(column, str):
        raise TypeError("column must be a string")
    if column not in df.columns:
        raise ValueError(f"column '{column}' not found in DataFrame")

    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def plot_relation(x: str, y: str, df: DataFrame, palette=None) -> None:
    """
    This function creates two plots: a scatter plot and a linear regression plot.
    The scatter plot shows the relationship between two variables, with points colored by year.
    The linear regression plot shows the relationship between the 'Happiness Score' and
    another variable with lines colored by year.

    Parameters
    ----------
    x (str): The name of the column to use for the x-axis in the plots.
    y (str): The name of the column to use for the y-axis in the scatter plot
    and for coloring lines in the linear regression plot.
    df (DataFrame): The DataFrame containing the data to plot.
    palette (str): The color palette to use for the plots.

    Returns
    ----------
    None
    """
    title = x + " vs " + y

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=x,
        y=y,
        data=df,
        hue="Year",
        palette=palette,
    )
    plt.title(title)
    plt.show()

    df["Year"] = df["Year"].astype("category")
    sns.lmplot(
        x=x,
        y=y,
        hue="Year",
        data=df,
        palette=palette,
        height=6,
        aspect=1.5,
    )
    plt.title(title)
    plt.show()
