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
