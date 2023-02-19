"""pytest fixtures """
import pandas as pd
import pytest


@pytest.fixture
def test_input_data_schema():
    """Generate schema dataframe to test schema modules"""

    schema = pd.DataFrame(
        {
            "pandas_dtype": {
                "col_boolean": "boolean",
                "col_integer": "integer",
                "col_string": "string",
                "col_floating": "floating",
                "col_mixed-integer-float": "floating",
                "col_mixed": "mixed",
                "col_MASK": "integer",
            },
            "numpy_dtype": {
                "col_boolean": float,
                "col_integer": float,
                "col_string": str,
                "col_floating": float,
                "col_mixed-integer-float": float,
                "col_mixed": str,
                "col_MASK": float,
            },
            "custom_dtype": {
                "col_boolean": "numeric",
                "col_integer": "numeric",
                "col_string": "nominal_str",
                "col_floating": "numeric",
                "col_mixed-integer-float": "numeric",
                "col_mixed": "lists",
                "col_MASK": "nominal_num",
            },
        }
    )
    return schema


@pytest.fixture
def test_schema_preprocessor():
    """Generate schema to test preprocessing.util.build_preprocessor"""
    schema = pd.DataFrame(
        {
            "custom_dtype": {
                "col_A0": "custom_type_A",
                "col_A1": "custom_type_A",
                "col_B0": "custom_type_B",
            }
        }
    )
    return schema


@pytest.fixture
def test_transformer_map():
    """Generate transformer_map to test preprocessing.util.build_preprocessor"""

    transformer_map = {
        "custom_type_A": "preprocessor_A",
        "custom_type_B": "preprocessor_B",
    }
    return transformer_map


@pytest.fixture
def test_preprocessor_steps():
    """Generate preprocessor_steps to test preprocessing.util.build_preprocessor"""

    preprocessor_steps_test = [
        ("column_transformers", None),
        ("feature_selector", "dummy_feature_selector"),
    ]

    return preprocessor_steps_test
