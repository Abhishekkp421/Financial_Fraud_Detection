import pandas as pd


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/transactions.csv")


print("===== ORIGINAL DATASET =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==========================================
# Missing Values
# ==========================================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


df = df.dropna()


# ==========================================
# Duplicate Records
# ==========================================

print("\n===== DUPLICATE RECORDS =====")
print("Duplicates:", df.duplicated().sum())


df = df.drop_duplicates()


# ==========================================
# FEATURE ENGINEERING
# ==========================================

# High transaction amount
df["high_amount"] = (
    df["amount"] > 5000
).astype(int)


# Night transaction
df["night_transaction"] = (
    (df["hour"] <= 5) |
    (df["hour"] >= 23)
).astype(int)


# Unusual distance
df["unusual_distance"] = (
    df["distance_km"] > 20
).astype(int)


# Low device trust
df["low_device_trust"] = (
    df["device_score"] < 0.40
).astype(int)


# High merchant risk
df["high_merchant_risk"] = (
    df["merchant_risk"] > 0.50
).astype(int)


# ==========================================
# Final Features
# ==========================================

features = [
    "amount",
    "hour",
    "distance_km",
    "device_score",
    "international",
    "merchant_risk",

    # Engineered features
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
# Display Results
# ==========================================

print("\n===== ENGINEERED FEATURES =====")

print(
    df[
        [
            "high_amount",
            "night_transaction",
            "unusual_distance",
            "low_device_trust",
            "high_merchant_risk"
        ]
    ].head()
)


print("\n===== FINAL FEATURES =====")
print(features)


print("\n===== TARGET DISTRIBUTION =====")
print(y.value_counts())


print("\n===== TARGET PERCENTAGE =====")
print(
    y.value_counts(normalize=True) * 100
)


print("\n===== FINAL DATASET SHAPE =====")
print(X.shape)


print("\n===== PREPROCESSING + FEATURE ENGINEERING COMPLETED =====")