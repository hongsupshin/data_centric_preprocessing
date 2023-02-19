"""Schema building functions for data preprocessing"""
from typing import Union

import numpy as np
import pandas as pd
from data_centric_preprocessing.util import log_and_raise
import logging

logger = logging.getLogger("data_centric_preprocessing")


def drop_non_informative_features(
    df: pd.DataFrame, seed_patterns: list
) -> pd.DataFrame:
    """Drop non-informative columns (seed, all-empty, and invariant)

    :param df: raw training data
    :param seed_patterns: list of regex patters that represent seed
    (=any columns where every value is unique like seed or index)
    :return: dataframe without the non-informative columns
    """
    cols_seed = df.columns[
        df.columns.astype(str).str.upper().str.contains("|".join(seed_patterns))
    ]
    df.drop(cols_seed, axis=1, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    df = df.loc[:, (df != df.iloc[0]).any()]

    return df


def build_schema(
    df: pd.DataFrame,
    numpy_dtype_map: dict[str, Union[type, None]],
    custom_dtype_map: dict[str, str],
    nominal_num_patterns: list[str] = [],
    catch_invariant_with_missing: bool = True,
) -> pd.DataFrame:
    """Build a schema from df by inferring pandas, numpy and custom dtypes

    :param df: input dataframe
    :param numpy_dtype_map: dict to map pd.api.types.infer_dtype output (str)
    to numpy type (general python type)
    :param custom_dtype_map: dict to map pd.api.types.infer_dtype output (str)
    to custom data types for column transformers in the preprocessing pipeline
    :param nominal_num_patterns: list of regex to detect "nominal_num" columns, defaults to None
    :param catch_invariant_with_missing: detect "invariant_with_missing" custom dtypes,
    defaults to True
    :raises ValueError: if numpy_dtype_map is missing
    :raises ValueError: if custom_dtype_map is missing
    :raises TypeError: if numpy_dtype_map is not a dict
    :raises TypeError: if custom_dtype_map is not a dict
    :return: schema as a dataframe
    """
    if not numpy_dtype_map:
        raise ValueError("numpy_dtype_map is missing.")
    if not custom_dtype_map:
        raise ValueError("custom_dtype_map is missing.")
    if not isinstance(numpy_dtype_map, dict):
        raise TypeError(f"numpy_dtype_map is {type(numpy_dtype_map)}, expected dict.")
    if not isinstance(custom_dtype_map, dict):
        raise TypeError(f"custom_dtype_map is {type(custom_dtype_map)}, expected dict.")

    pandas_dtype = df.apply(lambda x: pd.api.types.infer_dtype(x, skipna=True))

    schema = pd.DataFrame(pandas_dtype, columns=["pandas_dtype"])
    schema["numpy_dtype"] = schema["pandas_dtype"].map(numpy_dtype_map)
    schema["custom_dtype"] = schema["pandas_dtype"].map(custom_dtype_map)

    if nominal_num_patterns:
        # nominal-numeric columns refer to numeric columns w/o ordinality
        pattern = "|".join(nominal_num_patterns)
        cols_nominal_num = schema.index[
            schema.index.astype(str).str.upper().str.contains(pattern)
        ]
        schema.loc[cols_nominal_num, "custom_dtype"] = "nominal_num"

    if catch_invariant_with_missing:
        # when a column has at least one nan, (e.g., [0, 0, 0, nan, nan]), and
        # only two values remain after removing dups from it, (e.g., [0, nan])
        # the two values are nan & the single invariant value.
        # A column like this is referred to as "invariant with missing"
        df_with_na = df.loc[:, df.isna().sum() > 0]
        cols_invariant_with_missing = df_with_na.columns[
            df_with_na.apply(lambda s: len(s.drop_duplicates())) == 2
        ]
        schema.loc[
            cols_invariant_with_missing, "custom_dtype"
        ] = "invariant_with_missing"

    return schema


def cast_numpy_dtype(df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    """Cast numpy dtypes to df based on a inferred schema.

    The schema should have been already inferred from the df by using build_schema.
    This is strictly to get the raw input data ready for the downstream ML pipeline,
    NOT for resolving mismatch between training and serving.

    :param df: dataframe of a raw input dataset
    :param schema: schema inferred from df (build_schema output)
    :raises AssertionError: if columns do not match between df and schema
    :raises ValueError: if numpy dtype casting cannot be done (e.g., casting int to str values)
    :return: df with inferred numpy dtypes
    """
    if not np.array_equal(df.columns, schema["numpy_dtype"].index):
        log_and_raise(
            AssertionError,
            "Columns do not match between df and schema. "
            "Check which df is used to create the schema.",
        )

    try:
        schema_np = schema["numpy_dtype"]
        if "nominal_num" in schema["custom_dtype"].unique():
            cols_nominal_num = schema.index[schema["custom_dtype"] == "nominal_num"]
            schema_np.loc[cols_nominal_num] = str
        df_numpy = df.astype(dtype=schema_np)
    except ValueError as err:
        logger.exception(err)
        raise ValueError(
            "Cannot cast numpy dtypes to df. Check schema creation and numpy_dtype_map."
        ) from None

    return df_numpy
