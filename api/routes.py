from pathlib import Path

import joblib
import pandas as pd

from fastapi import APIRouter

from api.schemas import Customer

router = APIRouter()

MODEL = Path("data/models/churn_pipeline.pkl")

pipeline = joblib.load(MODEL)


@router.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "Running"
    }


@router.get("/model")
def model():

    return {
        "model": "Random Forest",
        "version": "1.0"
    }


@router.post("/predict")
def predict(customer: Customer):

    try:

        df = pd.DataFrame([customer.model_dump()])

        prediction = pipeline.predict(df)[0]

        probability = pipeline.predict_proba(df)[0][1]

        if probability >= 0.80:
            risk = "High"
        elif probability >= 0.50:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "prediction": (
                "Likely to Churn"
                if prediction == 1
                else "Not Likely to Churn"
            ),
            "probability": round(float(probability), 4),
            "risk_level": risk,
        }

    except Exception as e:

        return {
            "error": str(e)
        }