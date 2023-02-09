# Description

## Data challenges
Hardware verification data pose many challenges for ML. In addition to their high heterogeneity and dimensionality, the number and dtype of features change over time (e.g., float in the past later becomes string). Besides, raw data often have inaccurate dtypes (e.g., object for float), which requires type inference. Unfortunately, schemas are absent due to frequent changes in testing infrastructure and obscure feature meanings.

These issues can cause serious ML problems. Data preprocessing pipeline gets complicated, opaque, and inefficient, and becomes a major bottleneck in ML systems. The ML systems become more vulnerable to data leakage, bringing overoptimism in performance. Trained models frequently fail during serving due to feature set changes. Finally, without a good understanding of the data, non-standardized and hacky preprocessing methods can easily thrive.

To address these, I adopt a data-centric approach to build an ML pipeline. It is adaptive to data drifts, and increases modularity, transparency, and efficiency of data preprocessing. Besides, it utilizes standard python packages (pandas, numpy, and scikit-learn). This pipeline comes with these key features: bottom-up schema creation, schema-dependent data preprocessing, and serving-mismatch resolution. 

## Bottom-up schema creation
All features from the raw data have object dtypes although this is only true for a small subset. For instance, a numeric array of [0, 0, 1] is represented as object; array([0, 0, 1], dtype=object). Sklearn preprocessors can handle this, but true dtypes are still needed to apply correct preprocessing methods. For example, mean imputation strategy can’t be applied to string features. Besides, domain-specific custom dtypes need to be further identified by true dtypes, regex, et cetera. Correct inference of dtypes is a necessary step.

Among various dtype inference methods, I choose pandas.api.types.infer_dtype because of its granularity and capability to ignore null values. Since it returns a string label for a dtype (“string” for str), we map the label to numpy dtypes (e.g., {“string”: str}) for dtype casting. Finally, we map the labels to domain-specific custom types to build the data preprocessor pipeline. Feature names and their dtypes (pandas, numpy, and custom) are saved as a schema. Based on the numpy dtypes of the schema, dtypes of the raw data are corrected.

## Schema-dependent data preprocessor
Data preprocessing is done by a sklearn Pipeline (“preprocessor”) of pre-defined steps. The pipeline has a “column transformer” step to handle data heterogeneity. This step consists of multiple column transformers (sklearn ColumnTransformer). Each of them handles a feature subset of a specific custom dtype. A single column transformer requires a preprocessing method and a feature subset. The method comes from a look-up table (“transformer map”) and the feature names are loaded from the schema. Since the schema is inferred from the training data, the preprocessor structure is dependent on the contents of the data.

## Serving-mismatch resolution
During serving, the feature set is likely to have changed. Thus, we build a schema from the serving data and resolve feature-set and type mismatches by comparing it with the training schema. Unseen features are dropped, and missing features are added back as dummies whose dtypes come from the training schema. This simple resolution is designed to avoid complete failure due to frequent mismatches, not to ignore data drifts. As soon as the mismatches are detected, ML practitioners initiate manual data inspection.

## Data tuning
Verification data are highly domain-specific and abstract. Thus, a feature can have multiple interpretations. Since the data preprocessing and model pipelines can be merged, it’s possible to tune the preprocessing methods as if model hyperparameter tuning (i.e., “data tuning”). I tuned 16 preprocessing methods for 52 real-world benchmark datasets while fixing model hyperparameters (default values from a LightGBM classifier). Performance comparison between the best and the worst methods results in 0.11 AUROC on average across benchmark data. In other words, performance can be increased via data tuning without tuning model hyperparameters.

## Summary
This data-centric pipeline streamlines and automates the data preprocessing step, which often becomes a bottleneck in an ML system. How the system digests data reflects what we have in the raw data, and thus it is suitable for handling highly heterogeneous and frequently changing data. Besides, it allows data tuning, which can improve model performance significantly.
