"""Util function"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import logging
import warnings
from typing import Type, Union

logger = logging.getLogger("data_centric_preprocessing")


def log_and_raise(error_type: Type[Exception], message: str) -> None:
    """Shorthand utility for logging and raising an exception.

    :param error_type: error type (ValueError, TypeError, etc.)
    :param message: error message
    :raises error_type: if the condition is met, raise an error
    """
    try:
        raise error_type(message)
    except error_type as exc:
        logger.exception(exc)
        raise


def build_preprocessor(
    schema_training: pd.DataFrame, transformer_map: dict, preprocessor_steps: list
) -> sklearn.pipeline.Pipeline:
    """Build a data preprocessing pipeline.

    The preprocessor pipeline has a sklearn.compose.ColumnTransformer step.
    This step consists of multiple sklearn pipelines. Each pipeline corresponds to
    a custom dtype in the training schema inferred from an input dataset.

    :param schema_training: schema of an input training dataset
    :param transformer_map: dict of sklearn pipelines (preprocessing methods)
    :param preprocessor_steps: list of tuple pairs of ('step name', preprocessing method)
    :raises KeyError: if unknown data types are present in the schema
    :raises ValueError: if "column_transformers" is missing in the preprocessing_step
    :return: a sklearn pipeline of data preprocessor
    """
    known_custom_dtypes = set(schema_training["custom_dtype"])

    if known_custom_dtypes - (transformer_map.keys()):
        log_and_raise(
            KeyError, "Schema has unknown data types. Update the transformer map."
        )
    if not ("column_transformers" in [step[0] for step in preprocessor_steps]):
        log_and_raise(
            ValueError, "'column_transformers' is missing in preprocessor_step."
        )

    transformers = []
    for custom_dtype in sorted(known_custom_dtypes):
        cols = schema_training[
            schema_training["custom_dtype"] == custom_dtype
        ].index.values
        transformer = (custom_dtype, transformer_map[custom_dtype], cols)
        transformers.append(transformer)
    column_transformers = ColumnTransformer(transformers)
    preprocessor = Pipeline(preprocessor_steps)
    preprocessor.set_params(column_transformers=column_transformers)

    return preprocessor


def match_cols(
    X: pd.DataFrame, cols_fit: Union[pd.Index, np.ndarray, list]
) -> pd.DataFrame:
    """Matches columns in DataFrame with expected columns.

    Compares columns of X and cols_fit and
    1. drop the columns that were not seen during fitting from X
    2. add the columns that are missing in cols_fit back to X as missing

    :param X: data to transform
    :param cols_fit: columns that were seen during fitting
    :return: data that has the same set of columns as cols_fit
    """
    if len(cols_fit) == 0:
        log_and_raise(ValueError, "cols_fit is empty.")
    X = drop_cols_not_seen(X, cols_fit)
    X = add_missing_cols(X, cols_fit)
    # match the column order as same as cols_fit
    X = X[cols_fit]
    return X


def drop_cols_not_seen(
    X: pd.DataFrame, cols_fit: Union[pd.Index, np.ndarray, list]
) -> pd.DataFrame:
    """Drop columns not in cols_fit.

    :param X: DataFrame to transform
    :param cols_fit: columns that were seen during fitting
    :return: data with columns not in cols_fit dropped
    :raises: ValueError: If all columns in X were never seen during fitting.
    """
    cols_not_seen = set(X.columns) - set(cols_fit)
    # drop unseen columns
    if cols_not_seen == set(X.columns):
        log_and_raise(ValueError, "All columns in X were never seen during fitting.")

    if len(cols_not_seen) > 0:
        logger.warning(
            f"Columns {sorted(list(cols_not_seen))} were dropped because they are not in X."
        )
        X = X.drop(list(cols_not_seen), axis=1)
    return X


def add_missing_cols(
    X: pd.DataFrame, cols_fit: Union[pd.Index, np.ndarray, list]
) -> pd.DataFrame:
    """add the columns that are missing in cols_fit back to X as missing (nan values).

    :param X: DataFrame to transform
    :param cols_fit: columns that were seen during fitting
    :return: data with missing columns added with Nans
    """
    cols_missing = set(cols_fit) - set(X.columns)
    # add missing columns
    if len(cols_missing) > 0:
        logger.warning(
            f"Columns {sorted(list(cols_missing))} were added back. These contain only nan values. "
            f"These nan values will be handled by an imputer at a separate step."
        )
        X[list(cols_missing)] = np.ones((X.shape[0], len(cols_missing))) * np.nan
    return X


def replace_numpy_dtype_mismatch(
    schema_train: pd.DataFrame,
    schema_serve: pd.DataFrame,
    X_serve: pd.DataFrame,
) -> pd.DataFrame:
    """Replace any numpy dtype mismatches between X_serve and schema_train with nulls

    When these mismatches exist, trained models cannot digest X_serve.
    This function resolves them by replacing the mismatched columns with nulls.

    :param schema_train: training data schema
    :param schema_serve: serving data schema
    :param X_serve: serving dataframe which may have numpy dtype mismatch
    :return: X_serve dataframe whose mismatched columns have nans
    """
    if set(schema_train.index) != set(X_serve.columns):
        warnings.warn(
            "Column-mismatch between schema_train and X_serve "
            "resolved by match_cols()...",
            UserWarning,
        )
        X_serve = match_cols(X_serve, schema_train.index)

    cols_common = X_serve.columns.intersection(schema_serve.index)
    cols_mismatch = cols_common[
        schema_serve.loc[cols_common, "numpy_dtype"]
        != schema_train.loc[cols_common, "numpy_dtype"]
    ]

    if len(cols_mismatch) > 0:
        warnings.warn("Numpy dtype mismatches are replaced with nulls.", UserWarning)
        null_dummies = pd.DataFrame(np.nan, index=X_serve.index, columns=cols_mismatch)
        X_serve.drop(cols_mismatch, axis=1, inplace=True)
        X_serve = pd.concat([X_serve, null_dummies], axis=1)

    return X_serve[schema_train.index]
