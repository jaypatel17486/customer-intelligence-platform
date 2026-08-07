from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

MODEL = Path("data/models/churn_pipeline.pkl")
DATA = Path("data/processed/customer_churn_clean.csv")


def main():

    pipeline = joblib.load(MODEL)

    model = pipeline.named_steps["model"]

    preprocessor = pipeline.named_steps["preprocessor"]

    df = pd.read_csv(DATA)

    X = df.drop("Churn", axis=1)

    feature_names = preprocessor.get_feature_names_out()

    importance = model.feature_importances_

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )

    plt.title("Top 15 Most Important Features")

    plt.tight_layout()

    plt.savefig("results/images/feature_importance.png")

    print("Saved feature importance chart.")
    

if __name__ == "__main__":
    main()