import csv
import random
import math

random.seed(42)

rows = []

for _ in range(5000):

    # -----------------------------
    # Transaction features
    # -----------------------------

    amount = round(
        random.lognormvariate(math.log(1800), 1.0),
        2
    )

    hour = random.randint(0, 23)

    distance_km = round(
        max(0, random.gauss(8, 15)),
        2
    )

    device_score = round(
        min(1, max(0, random.betavariate(7, 2))),
        3
    )

    international = (
        1 if random.random() < 0.12 else 0
    )

    merchant_risk = round(
        random.betavariate(2, 5),
        3
    )


    # -----------------------------
    # Fraud risk factors
    # -----------------------------

    risk_score = -5.0

    # Very high transaction amount
    if amount > 10000:
        risk_score += 1.8
    elif amount > 5000:
        risk_score += 0.8

    # Unusual transaction time
    if hour <= 5:
        risk_score += 1.7
    elif hour >= 23:
        risk_score += 0.8

    # Unusual location
    if distance_km > 30:
        risk_score += 1.5
    elif distance_km > 15:
        risk_score += 0.6

    # Low device trust
    if device_score < 0.35:
        risk_score += 2.0
    elif device_score < 0.55:
        risk_score += 0.8

    # International transaction
    if international == 1:
        risk_score += 1.2

    # Risky merchant
    if merchant_risk > 0.60:
        risk_score += 2.0
    elif merchant_risk > 0.40:
        risk_score += 0.8


    # -----------------------------
    # Convert risk score to
    # probability
    # -----------------------------

    fraud_probability = 1 / (
        1 + math.exp(-risk_score)
    )


    # -----------------------------
    # Generate fraud label
    # -----------------------------

    is_fraud = (
        1
        if random.random() < fraud_probability
        else 0
    )


    rows.append([
        amount,
        hour,
        distance_km,
        device_score,
        international,
        merchant_risk,
        is_fraud
    ])


# -----------------------------
# Save dataset
# -----------------------------

file_path = "dataset/transactions.csv"

with open(
    file_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "amount",
        "hour",
        "distance_km",
        "device_score",
        "international",
        "merchant_risk",
        "is_fraud"
    ])

    writer.writerows(rows)


print("================================")
print("NEW DATASET GENERATED")
print("================================")

print("Total transactions:", len(rows))
print("File:", file_path)

fraud_count = sum(
    row[-1] for row in rows
)

legitimate_count = (
    len(rows) - fraud_count
)

print("Legitimate:", legitimate_count)
print("Fraudulent:", fraud_count)

print(
    "Fraud percentage:",
    round(
        fraud_count / len(rows) * 100,
        2
    ),
    "%"
)

print("\nDataset generation completed!")