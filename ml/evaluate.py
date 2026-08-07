from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/customer_churn_clean.csv")
MODEL = Path("data/models/churn_pipeline.pkl")

RESULTS = Path("results")
IMAGES = RESULTS / "images"


def main():

    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

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

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
    }

    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(RESULTS / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(IMAGES / "confusion_matrix.png")

    plt.close()

    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print()
    print("Files created:")
    print("results/metrics.json")
    print("results/classification_report.txt")
    print("results/images/confusion_matrix.png")


if __name__ == "__main__":
    main()