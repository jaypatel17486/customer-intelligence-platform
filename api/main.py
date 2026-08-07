from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

app.include_router(router)