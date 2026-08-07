from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.pipeline import build_preprocessor

DATA = Path("data/processed/customer_churn_clean.csv")

MODEL = Path("data/models/churn_pipeline.pkl")


def main():

    df = pd.read_csv(DATA)
    
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    preprocessor = build_preprocessor(df)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL,
    )

    print("Pipeline saved successfully!")


if __name__ == "__main__":
    main()