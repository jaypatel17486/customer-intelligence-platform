import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.xls")
PROCESSED_DATA = Path("data/processed/customer_churn_clean.csv")


def main():

    print("=" * 60)
    print("Customer Churn Data Preprocessing")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA)

    print(f"Original Shape: {df.shape}")

    # Remove customer ID
    df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    print("\nMissing Values:")
    print(df.isnull().sum())

    PROCESSED_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA,
        index=False,
    )

    print("\nProcessed Shape:", df.shape)

    print("\nDataset saved to:")
    print(PROCESSED_DATA)


if __name__ == "__main__":
    main()