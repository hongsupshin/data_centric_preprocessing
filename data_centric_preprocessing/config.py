"""Config for data preprocessing"""
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

transformer_map = {
    "invariant_with_missing": Pipeline(
        steps=[
            ("imputer", MissingIndicator(error_on_new=False)),
        ]
    ),
    "numeric": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    ),    
    "arrays": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder()),
        ]
    ),
    "nominal_str": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder()),
        ]
    ),    
    "nominal_num": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder()),
        ]
    ),    
}

preprocessor_steps = [
    ("column_transformers", None),
    ("feature_selection", VarianceThreshold(threshold=0.0)),
]