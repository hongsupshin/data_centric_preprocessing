"""Configs and constants"""

# pandas dtype to numpy dtype map
numpy_dtype_map = {
    "boolean": float,
    "integer": float,
    "string": str,
    "floating": float,
    "mixed-integer-float": float,
    "mixed-integer": float,
    "mixed": str,  # mixed types are arrays, which are treated as string
}

# pandas dtype to custom dtype map
custom_dtype_map = {
    "boolean": "numeric",
    "integer": "numeric",
    "string": "nominal_str",
    "floating": "numeric",
    "mixed-integer-float": "numeric",
    "mixed-integer": "numeric",
    "mixed": "arrays",  # mixed types are arrays, which are treated as string
}

# nominal numeric column regex patterns
# use pandas string operation to identify these columns from column names
# users can update the array if a new pattern needs to be added
nominal_num_patterns = [
    "PATTERN1$",
    "PATTERN2$",
]
