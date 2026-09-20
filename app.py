import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Model lives in models/best_model.pkl next to this file (override with MODEL_PATH)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

# Load trained ML model
model = joblib.load(MODEL_PATH)
CLASS_NAMES = list(model.named_steps["model"].classes_)

app = FastAPI(
    title="Order Return-Status Prediction API"
)


class OrderData(BaseModel):
    order_year: int
    order_month: int
    order_quarter: int
    brand: str
    subcategory: str
    quantity: int
    original_price_inr: float
    discounted_price_inr: float
    discount_percent: float
    final_amount_inr: float
    price_outlier_3sigma: bool
    price_outlier_IQR: bool
    is_festival_sale: bool
    festival_name: str
    is_prime_member: bool
    is_prime_eligible: bool
    payment_method: str
    delivery_type: str
    delivery_days: int
    product_rating: float
    product_weight_kg: float
    customer_age_group: str
    customer_city: str
    customer_state: str
    customer_tier: str
    customer_spending_tier: str
    customer_rating: float


@app.get("/")
def home():
    return {
        "message": "Order Return-Status Prediction API is running",
        "classes": CLASS_NAMES
    }


@app.post("/predict")
def predict(data: OrderData):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        predicted_class = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        class_probabilities = {
            cls: round(float(prob), 4)
            for cls, prob in zip(CLASS_NAMES, probabilities)
        }

        return {
            "predicted_return_status": predicted_class,
            "class_probabilities": class_probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
