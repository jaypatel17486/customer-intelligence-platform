import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(df):

    categorical = df.select_dtypes(include="object").columns.tolist()

    numerical = df.select_dtypes(exclude="object").columns.tolist()

    if "Churn" in categorical:
        categorical.remove("Churn")

    if "Churn" in numerical:
        numerical.remove("Churn")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical),
            ("num", numerical_transformer, numerical)
        ]
    )

    return preprocessor