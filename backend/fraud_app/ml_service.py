import joblib
import pandas as pd
from pathlib import Path


# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ML model path
MODEL_PATH = BASE_DIR / "ml_model" / "fraud_model.pkl"


# Load trained model
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]
threshold = model_data["threshold"]


def predict_fraud(transaction):
    """
    Predict whether a transaction is fraudulent.
    """

    # Feature engineering
    transaction["high_amount"] = (
        transaction["amount"] > 5000
    )

    transaction["night_transaction"] = (
        transaction["hour"] <= 5
        or transaction["hour"] >= 23
    )

    transaction["unusual_distance"] = (
        transaction["distance_km"] > 20
    )

    transaction["low_device_trust"] = (
        transaction["device_score"] < 0.40
    )

    transaction["high_merchant_risk"] = (
        transaction["merchant_risk"] > 0.50
    )

    # Convert to DataFrame
    input_data = pd.DataFrame(
        [transaction]
    )

    # Ensure correct feature order
    input_data = input_data[features]

    # Fraud probability
    probability = model.predict_proba(
        input_data
    )[0][1]

    # Final decision
    if probability >= threshold:
        prediction = "Fraudulent"
    else:
        prediction = "Legitimate"

    return {
        "fraud_probability": round(
            probability * 100, 2
        ),
        "prediction": prediction,
        "threshold": round(
            threshold * 100, 2
        )
    }