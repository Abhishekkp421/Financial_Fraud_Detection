import matplotlib

# Tkinter ko use nahi karna
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/transactions.csv")

# -------------------------------
# 1. Fraud Distribution
# -------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="is_fraud"
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraudulent"]
)

plt.tight_layout()
plt.savefig("fraud_distribution.png")
#plt.show()


# -------------------------------
# 2. Transaction Amount
# -------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="amount",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("amount_distribution.png")
#plt.show()


# -------------------------------
# 3. Fraud by Transaction Hour
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="hour",
    hue="is_fraud"
)

plt.title("Transactions by Hour")
plt.xlabel("Transaction Hour")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.savefig("fraud_by_hour.png")
#plt.show()


# -------------------------------
# 4. Distance vs Fraud
# -------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="is_fraud",
    y="distance_km"
)

plt.title("Distance vs Fraud")
plt.xlabel("Transaction Type")
plt.ylabel("Distance (km)")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraudulent"]
)

plt.tight_layout()
plt.savefig("distance_vs_fraud.png")
#plt.show()


# -------------------------------
# 5. Correlation Heatmap
# -------------------------------

plt.figure(figsize=(9, 7))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
#plt.show()


print("\nEDA visualization completed successfully!")

print("\nGenerated files:")
print("1. fraud_distribution.png")
print("2. amount_distribution.png")
print("3. fraud_by_hour.png")
print("4. distance_vs_fraud.png")
print("5. correlation_heatmap.png")