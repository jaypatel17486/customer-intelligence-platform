import joblib
import pandas as pd

from fastapi import APIRouter

from api.schemas import Customer

router = APIRouter()

pipeline = joblib.load("data/models/churn_pipeline.pkl")


@router.get("/")
def home():
    return {"message": "Customer Intelligence Platform API"}


@router.post("/predict")
def predict(customer: Customer):

    df = pd.DataFrame([customer.model_dump()])

    prediction = pipeline.predict(df)[0]

    probability = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": "Likely to Churn" if prediction == 1 else "Not Likely to Churn",
        "probability": round(float(probability), 4)
    }