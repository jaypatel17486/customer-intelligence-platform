from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


DATA = Path("data/processed/customer_churn_clean.csv")


def build_preprocessor(df):

    categorical = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = df.select_dtypes(exclude=["object", "string"]).columns.tolist()

    if "Churn" in categorical:
        categorical.remove("Churn")

    if "Churn" in numerical:
        numerical.remove("Churn")

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numerical,
            ),
        ]
    )


def evaluate(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
    }


def main():

    df = pd.read_csv(DATA)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = [
        (
            "Logistic Regression",
            Pipeline([
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("prep", preprocessor),
                ("model", RandomForestClassifier(random_state=42)),
            ]),
        ),
        (
            "XGBoost",
            Pipeline([
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]),
        ),
    ]

    results = []

    for name, model in models:
        results.append(
            evaluate(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    results = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(results)
    
    results.to_csv(
    "results/model_comparison.csv",
    index=False,
)

print("\nModel comparison saved to:")
print("results/model_comparison.csv")


if __name__ == "__main__":
    main()