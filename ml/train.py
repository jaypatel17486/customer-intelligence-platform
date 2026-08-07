from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

INPUT = Path("data/processed/customer_churn_features.csv")
MODEL = Path("data/models/churn_model.pkl")


def main():

    print("=" * 60)
    print("Customer Churn Model Training")
    print("=" * 60)

    df = pd.read_csv(INPUT)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}\n")

    print(classification_report(y_test, predictions))

    MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL)

    print("=" * 60)
    print("Model saved successfully!")
    print(MODEL)
    print("=" * 60)


if __name__ == "__main__":
    main()