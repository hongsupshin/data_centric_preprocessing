"""Test util.py functions"""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from data_centric_preprocessing.util import (
    build_preprocessor,
    replace_numpy_dtype_mismatch,
)


def test_build_preprocessor(
    test_schema_preprocessor: pd.DataFrame,
    test_transformer_map: dict,
    test_preprocessor_steps: list,
) -> None:
    """Test if test_build_preprocessor returns desired pipeline steps

    Since it's difficult to compare class instances, we compare the pipeline structure
    using pipeline.get_params
    1. step names: name of each preprocessing step (column_transformers, feature_selector)
    2. step values
        2.1. column_transformers: custom_type, preprocessor, and column names (e.g., col_A0)
        2.2. feature_selector: "dummy_feature_selector" as defined in preprocessor_steps_test

    Given:
        schema, transformer_map, preprocessor_steps, and a dataframe to transform
    When:
        build_preprocessor is called
    Then:
        return preprocessor with column transformers
    """

    preprocessor = build_preprocessor(
        test_schema_preprocessor, test_transformer_map, test_preprocessor_steps
    )

    preprocessor_expected = Pipeline(
        steps=[
            (
                "column_transformers",
                ColumnTransformer(
                    transformers=[
                        (
                            "custom_type_A",
                            "preprocessor_A",
                            np.array(["col_A0", "col_A1"], dtype=object),
                        ),
                        (
                            "custom_type_B",
                            "preprocessor_B",
                            np.array(["col_B0"], dtype=object),
                        ),
                    ]
                ),
            ),
            ("feature_selector", "dummy_feature_selector"),
        ]
    )

    # step name check: 2 steps exist (column_transformers, feature_selector)
    assert [step[0] for step in preprocessor.get_params()["steps"]] == [
        step[0] for step in preprocessor_expected.get_params()["steps"]
    ]

    # step 1 (column_transformers) check
    transformers = np.array(
        preprocessor.get_params()["column_transformers__transformers"], dtype=object
    )
    transformers_expected = np.array(
        preprocessor_expected.get_params()["column_transformers__transformers"],
        dtype=object,
    )
    for transformer, transformer_expected in zip(transformers, transformers_expected):
        custom_type, preprocessor_, col_names = transformer
        (
            custom_type_expected,
            preprocessor_expected_,
            col_names_expected,
        ) = transformer_expected
        assert np.array_equal(custom_type, custom_type_expected)
        assert np.array_equal(preprocessor_, preprocessor_expected_)
        assert np.array_equal(col_names, col_names_expected)

    # step 2 (feature_selector) check
    assert (
        preprocessor.get_params()["feature_selector"]
        == preprocessor_expected.get_params()["feature_selector"]
    )


def test_build_preprocessor_fail_unknown_dtypes(
    test_schema_preprocessor: pd.DataFrame,
    test_transformer_map: dict,
    test_preprocessor_steps: list,
) -> None:
    """Test test_build_preprocessor raises KeyError if unknown dtypes are in the schema

    Given:
        input data schema with unknown custom dtypes
    When:
        build_preprocessor is called
    Then:
        raises KeyError
    """
    transformer_map_missing = test_transformer_map.copy()
    del transformer_map_missing["custom_type_A"]

    with pytest.raises(
        KeyError, match="Schema has unknown data types. Update the transformer map."
    ):
        build_preprocessor(
            test_schema_preprocessor, transformer_map_missing, test_preprocessor_steps
        )


def test_build_preprocessor_fail_missing_column_transformer(
    test_schema_preprocessor: pd.DataFrame,
    test_transformer_map: dict,
) -> None:
    """Test test_build_preprocessor raises ValueError
    if column_transformers is missing in preprocessing_step

    Given:
        steps with missing column transformer
    When:
        build_preprocessor is called
    Then:
        raises ValueError
    """
    preprocessor_steps_missing = [("feature_selector", None)]

    with pytest.raises(
        ValueError, match="'column_transformers' is missing in preprocessor_step."
    ):
        build_preprocessor(
            test_schema_preprocessor, test_transformer_map, preprocessor_steps_missing
        )


def test_replace_numpy_dtype_mismatch():
    """Test replace_numpy_dtype_mismatch

    Given:
        schema_train, schema_serve, and X_serve
    When:
        replace_numpy_dtype_mismatch is called
    Then:
        return X_serve with nan dummies for mismatched columns
    """
    schema_train = pd.DataFrame(
        {
            "numpy_dtype": {"A": float, "B": str, "C": float},
        }
    )
    schema_serve = pd.DataFrame(
        {
            # Column "B" is missing in X_serve -> a nan column will be added back
            # Column "C" now has str dtype -> a nan column will replace existing values
            "numpy_dtype": {"A": float, "C": str, "D": str},
        }
    )
    X_serve = pd.DataFrame(
        {
            "A": {0: 0, 1: 1.0, 2: np.nan},
            "C": {0: "a", 1: "b", 2: np.nan},
            "D": {0: "c", 1: "d", 2: np.nan},
        },
    )
    expected = pd.DataFrame(
        {
            "A": {0: 0.0, 1: 1.0, 2: np.nan},
            "B": {0: np.nan, 1: np.nan, 2: np.nan},
            "C": {0: np.nan, 1: np.nan, 2: np.nan},
        }
    )
    output = replace_numpy_dtype_mismatch(schema_train, schema_serve, X_serve)
    pd.testing.assert_frame_equal(output, expected)


@pytest.mark.parametrize(
    "schema_serve, X_serve, warning_msg",
    [
        (
            pd.DataFrame({"numpy_dtype": {"A": float, "C": str}}),
            pd.DataFrame({"A": {0: 0}, "C": {0: "a"}}),
            "Column-mismatch between schema_train and X_serve "
            "resolved by match_cols()...",
        ),
        (
            pd.DataFrame({"numpy_dtype": {"A": float, "B": float}}),
            pd.DataFrame({"A": {0: 0}, "B": {0: 0}}),
            "Numpy dtype mismatches are replaced with nulls.",
        ),
    ],
)
def test_replace_numpy_dtype_mismatch_warnings(
    schema_serve: pd.DataFrame, X_serve: pd.DataFrame, warning_msg: str
) -> None:
    """Test replace_numpy_dtype_mismatch warnings

    Given:
        schema_train, schema_serve, and X_serve
    When:
        replace_numpy_dtype_mismatch is called
    Then:
        return X_serve with nan dummies for mismatched columns
    """
    schema_train = pd.DataFrame({"numpy_dtype": {"A": float, "B": str}})
    with pytest.warns(UserWarning, match=warning_msg):
        replace_numpy_dtype_mismatch(schema_train, schema_serve, X_serve)
