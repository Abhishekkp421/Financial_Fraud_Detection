import joblib
import pandas as pd


# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================

model_data = joblib.load(
    "ml_model/fraud_model.pkl"
)

model = model_data["model"]
features = model_data["features"]
threshold = model_data["threshold"]


# ==========================================
# 2. TEST TRANSACTION
# ==========================================

transaction = {

    "amount": 800,

    "hour": 14,

    "distance_km": 3,

    "device_score": 0.90,

    "international": 0,

    "merchant_risk": 0.10
}


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

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


# Convert True/False to 0/1
for key in [
    "high_amount",
    "night_transaction",
    "unusual_distance",
    "low_device_trust",
    "high_merchant_risk"
]:
    transaction[key] = int(transaction[key])


# ==========================================
# 4. CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame(
    [transaction],
    columns=features
)


# ==========================================
# 5. PREDICT FRAUD PROBABILITY
# ==========================================

probability = model.predict_proba(
    input_data
)[0][1]


prediction = (
    "Fraudulent"
    if probability >= threshold
    else "Legitimate"
)


# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("\n================================")
print("FRAUD DETECTION RESULT")
print("================================")

print(
    "Fraud Probability:",
    round(probability * 100, 2),
    "%"
)

print(
    "Decision:",
    prediction
)

print(
    "Threshold:",
    threshold
)

print("\nPrediction completed successfully!")