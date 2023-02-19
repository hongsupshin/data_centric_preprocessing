"""Test create_schema.py functions"""
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data_centric_preprocessing.create_schema import (
    build_schema,
    cast_numpy_dtype,
    drop_non_informative_features,
)
from data_centric_preprocessing.data_schema import (
    custom_dtype_map,
    nominal_num_patterns,
    numpy_dtype_map,
)


def test_drop_non_informative_features():
    """Test drop_non_informative_features.

    Given:
        A dataframe containing informative and non-informative columns
    When:
        drop_non_informative_features is called
    Then:
        return a dataframe without the non-informative columns
    """
    data = pd.DataFrame(
        {
            "colseed": [0, 1, 2],
            "col1": [None, None, None],
            "col2": [np.nan, np.nan, np.nan],
            "col3": [None, np.nan, None],
            "col4": [0.0, 0.0, 0.0],
            "col5": ["foo", "foo", "foo"],
            "col6": [True, True, True],
            "col7": [0, 1, 2],
            "col8": [np.nan, np.nan, 1],  # invariant with missing
        }
    )

    transformed = drop_non_informative_features(data, ["SEED$"])
    expected_transformed = pd.DataFrame(
        {
            "col7": [0, 1, 2],
            "col8": [np.nan, np.nan, 1],  # invariant with missing survives
        }
    )
    pd.testing.assert_frame_equal(transformed, expected_transformed)


@pytest.mark.parametrize(
    "data, expected_pandas_dtype, expected_numpy_dtype, expected_custom_dtype, \
    catch_invariant_with_missing",
    [
        ({0: [None, True, False]}, "boolean", float, "numeric", True),
        ({0: [None, True, False]}, "boolean", float, "numeric", False),
        ({0: [None, 0, 1.0]}, "floating", float, "numeric", True),
        ({0: [None, 0, 1.0]}, "floating", float, "numeric", False),
        ({0: [[0], [1], [2]]}, "mixed", str, "arrays", True),
        ({0: [[0], [1], [2]]}, "mixed", str, "arrays", False),
        ({0: [None, [0], [1]]}, "mixed", str, "arrays", True),
        ({0: [None, [0], [1]]}, "mixed", str, "arrays", False),
        ({0: [None, [], [0]]}, "mixed", str, "arrays", True),
        ({0: [None, [], [0]]}, "mixed", str, "arrays", False),
        ({0: [None, "a", "b"]}, "string", str, "nominal_str", True),
        ({0: [None, "a", "b"]}, "string", str, "nominal_str", False),
        ({0: [None, 0, 1]}, "floating", float, "numeric", True),
        ({0: [None, 0, 1]}, "floating", float, "numeric", False),
        ({"fooPATTERN1": [None, 0.0, 1.0]}, "floating", float, "nominal_num", True),
        ({"fooPATTERN1": [None, 0.0, 1.0]}, "floating", float, "nominal_num", False),
        ({"fooPATTERN2": [None, 0.0, 1.0]}, "floating", float, "nominal_num", True),
        ({"fooPATTERN2": [None, 0.0, 1.0]}, "floating", float, "nominal_num", False),
        ({0: [None, True, True]}, "boolean", float, "invariant_with_missing", True),
        ({0: [None, True, True]}, "boolean", float, "numeric", False),
        ({0: [None, [0], [0]]}, "mixed", str, "invariant_with_missing", True),
        ({0: [None, [0], [0]]}, "mixed", str, "arrays", False),
        ({0: [None, 0.0, 0]}, "floating", float, "invariant_with_missing", True),
        ({0: [None, 0.0, 0]}, "floating", float, "numeric", False),
        ({0: [None, "a", "a"]}, "string", str, "invariant_with_missing", True),
        ({0: [None, "a", "a"]}, "string", str, "nominal_str", False),
    ],
)
def test_build_schema_dtype_inference(
    data: dict,
    expected_pandas_dtype: str,
    expected_numpy_dtype: type,
    expected_custom_dtype: str,
    catch_invariant_with_missing: bool,
) -> None:
    """Test build_schema's dype inference functionality.
    Given:
        A data array with various data types
    When:
        build_schema is called
    Then:
        return a schema (dataframe) with pandas, numpy, and custom data types
    """

    schema = build_schema(
        pd.DataFrame(data),
        numpy_dtype_map,
        custom_dtype_map,
        nominal_num_patterns,
        catch_invariant_with_missing,
    )
    assert schema["pandas_dtype"].values == expected_pandas_dtype
    assert schema["numpy_dtype"].values == expected_numpy_dtype
    assert schema["custom_dtype"].values == expected_custom_dtype


def test_build_schema():
    """Test build_schema.
    Given:
        A data frame with multiple columns
    When:
        build_schema is called
    Then:
        return a schema (dataframe) with pandas, numpy, and custom data types
    """

    data = pd.DataFrame(
        {
            "bool": [None, True, False],
            "mixed": [[0], [1], [2]],
            "fooPATTERN1": [None, 0.0, 1.0],
            "fooPATTERN2": [None, 0, 1],
            "invariant_with_missing": [None, 0, 0],
        }
    )

    schema = build_schema(
        data,
        numpy_dtype_map,
        custom_dtype_map,
        nominal_num_patterns,
        True,
    )

    schema_expected = pd.DataFrame(
        {
            "pandas_dtype": {
                "bool": "boolean",
                "mixed": "mixed",
                "fooPATTERN1": "floating",
                "fooPATTERN2": "floating",
                "invariant_with_missing": "floating",
            },
            "numpy_dtype": {
                "bool": float,
                "mixed": str,
                "fooPATTERN1": float,
                "fooPATTERN2": float,
                "invariant_with_missing": float,
            },
            "custom_dtype": {
                "bool": "numeric",
                "mixed": "arrays",
                "fooPATTERN1": "nominal_num",
                "fooPATTERN2": "nominal_num",
                "invariant_with_missing": "invariant_with_missing",
            },
        }
    )
    pd.testing.assert_frame_equal(schema, schema_expected)


@pytest.mark.parametrize(
    "numpy_dtype_map_input, custom_dtype_map_input, exception, error_msg",
    [
        (None, custom_dtype_map, ValueError, "numpy_dtype_map is missing."),
        ({}, custom_dtype_map, ValueError, "numpy_dtype_map is missing."),
        (numpy_dtype_map, None, ValueError, "custom_dtype_map is missing."),
        (numpy_dtype_map, {}, ValueError, "custom_dtype_map is missing."),
        (
            1,
            custom_dtype_map,
            TypeError,
            "numpy_dtype_map is <class 'int'>, expected dict.",
        ),
        (
            numpy_dtype_map,
            1,
            TypeError,
            "custom_dtype_map is <class 'int'>, expected dict.",
        ),
    ],
)
def test_build_schema_fails(
    numpy_dtype_map_input: Any,
    custom_dtype_map_input: Any,
    exception: type,
    error_msg: str,
) -> None:
    """Test build_schema() raises an error due to bad input."""
    data = pd.DataFrame({0: [1, 2, 3]})
    with pytest.raises(exception, match=error_msg):
        build_schema(
            data,
            numpy_dtype_map_input,
            custom_dtype_map_input,
            nominal_num_patterns=[],
            catch_invariant_with_missing=True,
        )


def test_cast_numpy_dtype(test_input_data_schema):
    """Test cast_numpy_dtype.

    Given:
        A dataframe of an input dataset (data types are not correctly inferred)
    When:
        _cast_numpy_dtype is called
    Then:
        return a dataframe with the correct data types based on schema
    """

    test_raw_input_data = pd.DataFrame(
        {
            "col_boolean": [True, False],
            "col_integer": [0, 1],
            "col_string": ["a", "b"],
            "col_floating": [0.0, 1.0],
            "col_mixed-integer-float": [0, 1.0],
            "col_mixed": [[0], [0, 1]],
            "col_MASK": [0, 1],
        }
    )

    transformed = cast_numpy_dtype(
        df=test_raw_input_data, schema=test_input_data_schema
    )
    expected_transformed = pd.DataFrame(
        {
            "col_boolean": {0: 1.0, 1: 0.0},
            "col_integer": {0: 0.0, 1: 1.0},
            "col_string": {0: "a", 1: "b"},
            "col_floating": {0: 0.0, 1: 1.0},
            "col_mixed-integer-float": {0: 0.0, 1: 1.0},
            "col_mixed": {0: "[0]", 1: "[0, 1]"},
            "col_MASK": {0: "0", 1: "1"},
        }
    )
    pd.testing.assert_frame_equal(transformed, expected_transformed)


def test_cast_numpy_dtype_no_nominal_num(test_input_data_schema):
    """Test cast_numpy_dtype when nominal num is absent.

    Given:
        A dataframe of an input dataset (data types are not correctly inferred)
    When:
        _cast_numpy_dtype is called
    Then:
        return a dataframe with the correct data types based on schema
    """

    test_raw_input_data = pd.DataFrame(
        {
            "col_boolean": [True, False],
            "col_integer": [0, 1],
            "col_string": ["a", "b"],
            "col_floating": [0.0, 1.0],
            "col_mixed-integer-float": [0, 1.0],
            "col_mixed": [[0], [0, 1]],
        }
    )

    schema = test_input_data_schema
    schema.drop(index="col_MASK", inplace=True)
    transformed = cast_numpy_dtype(df=test_raw_input_data, schema=schema)
    expected_transformed = pd.DataFrame(
        {
            "col_boolean": {0: 1.0, 1: 0.0},
            "col_integer": {0: 0.0, 1: 1.0},
            "col_string": {0: "a", 1: "b"},
            "col_floating": {0: 0.0, 1: 1.0},
            "col_mixed-integer-float": {0: 0.0, 1: 1.0},
            "col_mixed": {0: "[0]", 1: "[0, 1]"},
        }
    )
    pd.testing.assert_frame_equal(transformed, expected_transformed)


@pytest.mark.parametrize(
    "df_input, exception, error_msg",
    [
        (  # "col_mixed is missing"
            pd.DataFrame(
                {
                    "col_boolean": [True, False],
                    "col_integer": [0, 1],
                    "col_string": ["a", "b"],
                    "col_floating": [0.0, 1.0],
                    "col_mixed-integer-float": [0, 1.0],
                }
            ),
            AssertionError,
            "Columns do not match between df and schema. "
            "Check which df is used to create the schema.",
        ),
        (  # "cannot cast float to col_boolean"
            pd.DataFrame(
                {
                    "col_boolean": ["a", "b"],
                    "col_integer": [0, 1],
                    "col_string": ["a", "b"],
                    "col_floating": [0.0, 1.0],
                    "col_mixed-integer-float": [0, 1.0],
                    "col_mixed": [[0], [0, 1]],
                    "col_MASK": [0, 1],
                }
            ),
            ValueError,
            "Cannot cast numpy dtypes to df. Check schema creation and numpy_dtype_map.",
        ),
    ],
)
def test_cast_numpy_dtype_fails(
    df_input: pd.DataFrame,
    test_input_data_schema: pd.DataFrame,
    exception: type,
    error_msg: str,
) -> None:
    """Test _cast_numpy_dtype() raises an error due to bad input.

    Given:
        Case 0: A dataframe with a column missing compared to schema's columns
        Case 1: A dataframe where a column has string values but
        schema's inferred data type is float
    When:
        cast_numpy_dtype is called
    Then:
        Case 0: raises AssertionError
        Case 1: raise ValueError
    """

    with pytest.raises(exception, match=error_msg):
        cast_numpy_dtype(df_input, test_input_data_schema)
