# AI-Based Financial Fraud Detection System

An end-to-end machine learning based web application for detecting potentially fraudulent financial transactions.

## Project Overview

The project analyzes financial transaction details and predicts whether a transaction is potentially fraudulent or legitimate.

The system uses Machine Learning for fraud prediction and Django for developing the web application.

## Key Features

- User Registration and Login
- Financial Transaction Prediction
- Fraud Probability Calculation
- Low, Medium and High Risk Levels
- Fraud Alert for Suspicious Transactions
- User Transaction History
- User Dashboard
- Fraud Analysis Chart
- Admin Dashboard
- Admin All-Users Transaction History
- CSV Bulk Transaction Upload
- Automatic Fraud Prediction for CSV Records
- Multiple Machine Learning Model Comparison
- IST Timezone Support

## Machine Learning

### Input Features

The model uses the following transaction features:

- Amount
- Transaction Hour
- Distance in Kilometers
- Device Score
- International Transaction
- Merchant Risk

### Feature Engineering

Additional features are created from the transaction data:

- High Amount
- Night Transaction
- Unusual Distance
- Low Device Trust
- High Merchant Risk

### Models Compared

The following classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

XGBoost was selected as the final model based on the selected F1-score and precision criterion after threshold tuning.

## Model Evaluation

The models were evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC

Accuracy was not used as the only evaluation metric because fraud detection is an imbalanced classification problem.

## Technology Stack

### Programming Language
- Python

### Machine Learning
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

### Data Visualization
- Matplotlib
- Seaborn
- Chart.js

### Web Development
- Django
- HTML
- CSS
- JavaScript

### Database
- SQLite

### Version Control
- Git
- GitHub

## Project Structure

```text
Financial_Fraud_Detection/
│
├── dataset/
│   ├── transactions.csv
│   └── test_transactions.csv
│
├── ml_model/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── save_model.py
│   ├── eda.py
│   ├── eda_visualization.py
│   └── fraud_model.pkl
│
├── backend/
│   ├── manage.py
│   ├── fraud_detection/
│   │   ├── settings.py
│   │   └── urls.py
│   │
│   └── fraud_app/
│       ├── models.py
│       ├── views.py
│       ├── ml_service.py
│       ├── admin.py
│       └── templates/
│
├── venv/
│
└── README.md
## Application Screenshots

### Home - Fraud Detection

The home page allows users to enter transaction details and receive an ML-based fraud prediction.

### User Dashboard

The user dashboard displays transaction statistics, fraud/legitimate counts and fraud analysis through charts.

### Transaction History

Users can view their previous transaction predictions, fraud probability and risk level.

### Admin Dashboard

The admin dashboard provides overall transaction statistics, fraud analysis and recent transaction monitoring.

### Admin Transaction History

Admins can view transaction records across all users.

### CSV Bulk Upload

Admins can upload a CSV file and perform fraud prediction on multiple transactions.