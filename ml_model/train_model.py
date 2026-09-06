import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("dataset/transactions.csv")

print("================================")
print("DATASET")
print("================================")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================

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


# =====================================================
# 3. FEATURES AND TARGET
# =====================================================

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


print("\n================================")
print("FEATURES")
print("================================")

print("Total features:", len(features))

for feature in features:
    print("-", feature)


# =====================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


print("\n================================")
print("DATA SPLIT")
print("================================")

print("Training samples   :", len(X_train))
print("Validation samples :", len(X_validation))
print("Testing samples    :", len(X_test))


# =====================================================
# 5. DEFINE ML MODELS
# =====================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]),


    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),


    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),


    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )
}


# =====================================================
# 6. THRESHOLD SEARCH FUNCTION
# =====================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = 0

    thresholds = np.arange(
        0.10,
        0.51,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# =====================================================
# 7. TRAIN + EVALUATE
# =====================================================

results = []


for name, model in models.items():

    print("\n================================")
    print("TRAINING:", name)
    print("================================")


    # Train model
    model.fit(
        X_train,
        y_train
    )


    # -------------------------------------------------
    # Validation prediction
    # -------------------------------------------------

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]


    # Find threshold
    best_threshold, validation_f1 = (
        find_best_threshold(
            y_validation,
            validation_probability
        )
    )


    print(
        "Best Threshold:",
        round(best_threshold, 2)
    )

    print(
        "Validation F1:",
        round(validation_f1, 4)
    )


    # -------------------------------------------------
    # Test prediction
    # -------------------------------------------------

    test_probability = model.predict_proba(
        X_test
    )[:, 1]


    test_prediction = (
        test_probability >= best_threshold
    ).astype(int)


    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probability
    )

    cm = confusion_matrix(
        y_test,
        test_prediction
    )


    # -------------------------------------------------
    # Print metrics
    # -------------------------------------------------

    print("\nTEST RESULTS")

    print(
        "Precision :",
        round(precision, 4)
    )

    print(
        "Recall    :",
        round(recall, 4)
    )

    print(
        "F1-Score  :",
        round(f1, 4)
    )

    print(
        "ROC-AUC   :",
        round(roc_auc, 4)
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)


    results.append({

        "model": name,

        "threshold": best_threshold,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc

    })


# =====================================================
# 8. MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)


print("\n\n================================")
print("FINAL MODEL COMPARISON")
print("================================")

print(
    results_df.to_string(
        index=False
    )
)


# =====================================================
# 9. BEST MODEL
# =====================================================

best_model = results_df.iloc[0]


print("\n================================")
print("BEST MODEL")
print("================================")

print(
    "Model     :",
    best_model["model"]
)

print(
    "Threshold :",
    round(
        best_model["threshold"],
        2
    )
)

print(
    "Precision :",
    round(
        best_model["precision"],
        4
    )
)

print(
    "Recall    :",
    round(
        best_model["recall"],
        4
    )
)

print(
    "F1-Score  :",
    round(
        best_model["f1_score"],
        4
    )
)

print(
    "ROC-AUC   :",
    round(
        best_model["roc_auc"],
        4
    )
)


print("\n================================")
print("FEATURE ENGINEERING + TRAINING COMPLETED")
print("================================")