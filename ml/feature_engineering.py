import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

INPUT = Path("data/processed/customer_churn_clean.csv")
OUTPUT = Path("data/processed/customer_churn_features.csv")


def main():

    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    df = pd.read_csv(INPUT)

    # Binary Columns
    binary_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]

    encoder = LabelEncoder()

    for column in binary_columns:
        df[column] = encoder.fit_transform(df[column])

    # One-Hot Encode
    categorical_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print(df.head())

    print("\nShape:", df.shape)

    print("\nSaved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()