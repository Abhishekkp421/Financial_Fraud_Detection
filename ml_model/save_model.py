import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/transactions.csv")


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

df["high_amount"] = (
    df["amount"] > 5000
).astype(int)

df["night_transaction"] = (
    (df["hour"] <= 5) |
    (df["hour"] >= 23)
).astype(int)

df["unusual_distance"] = (
    df["distance_km"] > 20
).astype(int)

df["low_device_trust"] = (
    df["device_score"] < 0.40
).astype(int)

df["high_merchant_risk"] = (
    df["merchant_risk"] > 0.50
).astype(int)


# ==========================================
# 3. FEATURES
# ==========================================

features = [
    "amount",
    "hour",
    "distance_km",
    "device_score",
    "international",
    "merchant_risk",
    "high_amount",
    "night_transaction",
    "unusual_distance",
    "low_device_trust",
    "high_merchant_risk"
]

target = "is_fraud"


X = df[features]
y = df[target]


# ==========================================
# 4. TRAIN MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)


print("Training XGBoost model...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# 5. SAVE MODEL INFORMATION
# ==========================================

model_data = {

    "model": model,

    "features": features,

    "threshold": 0.18
}


joblib.dump(
    model_data,
    "ml_model/fraud_model.pkl"
)


# ==========================================
# 6. CONFIRMATION
# ==========================================

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print(
    "Model     : XGBoost"
)

print(
    "Features  :",
    len(features)
)

print(
    "Threshold :",
    0.18
)

print(
    "File      : ml_model/fraud_model.pkl"
)

print("\nModel ready for prediction!")