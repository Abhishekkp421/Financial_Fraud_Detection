import pandas as pd

# Dataset load
df = pd.read_csv("dataset/transactions.csv")

# Dataset ka size
print("\n===== DATASET SHAPE =====")
print(df.shape)

# First 5 records
print("\n===== FIRST 5 RECORDS =====")
print(df.head())

# Column names
print("\n===== COLUMNS =====")
print(df.columns.tolist())

# Data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Duplicate rows
print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

# Fraud distribution
print("\n===== FRAUD DISTRIBUTION =====")
print(df["is_fraud"].value_counts())

# Fraud percentage
print("\n===== FRAUD PERCENTAGE =====")
print(df["is_fraud"].value_counts(normalize=True) * 100)

# Statistical summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())