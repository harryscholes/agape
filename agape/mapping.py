"""Map gene IDs to other database IDs or gene names.
"""
import os
import pandas as pd
from typing import Dict

__all__ = ["gene2symbol"]

data = os.environ["AGAPEDATA"]


def dictify(df, key: str, value: str) -> dict:
    """Builds a dictionary using two columns of a Pandas DataFrame.

    # Arguments
        df: DataFrame, dataframe
        key: str, column name
        value: str, column name

    # Returns
        dict: key value mapping

    >>> dictify(pd.DataFrame({"A": [0, 1], "B": ["x", "y"]}), "A", "B")
    {0: 'x', 1: 'y'}
    """
    try:
        return {k: v for k, v in zip(df[key], df[value])}
    except KeyError as err:
        err.args = ((f"`{err.args[0]}` is not a valid key or value.",
                     f"Must be in: {set(df.columns)}"))
        raise


def gene2symbol(key: str, value: str) -> Dict[str, str]:
    """Map between S. pombe gene IDs, symbols, synonyms and names.

    # Arguments
        key: str, one of {"ID", "Symbol", "Synonym", "Name"}
        value: str, one of {"ID", "Symbol", "Synonym", "Name"}

    # Returns
        dict: key value mapping
    """
    df = pd.read_csv(os.path.join(data, "sysID2product.tsv"),
                     skiprows=1,
                     header=None,
                     sep="\t")
    df.columns = ["ID", "Symbol", "Synonymns", "Name"]
    return dictify(df, key, value)
