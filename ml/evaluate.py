import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/customer_churn_clean.csv")
MODEL = Path("data/models/churn_pipeline.pkl")


def main():

    df = pd.read_csv(DATA)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = joblib.load(MODEL)

    predictions = pipeline.predict(X_test)

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()