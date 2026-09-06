import pandas as pd
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from .ml_service import predict_fraud
from .models import Transaction

@login_required
def home(request):
    result = None

    if request.method == "POST":

        transaction = {
            "amount": float(request.POST.get("amount")),
            "hour": int(request.POST.get("hour")),
            "distance_km": float(request.POST.get("distance_km")),
            "device_score": float(request.POST.get("device_score")),
            "international": int(
                request.POST.get("international")
            ),
            "merchant_risk": float(
                request.POST.get("merchant_risk")
            )
        }

        result = predict_fraud(transaction)
        probability = result["fraud_probability"]

        if probability >= 70:
         result["risk_level"] = "High Risk"
        elif probability >= 30:
         result["risk_level"] = "Medium Risk"
        else:
         result["risk_level"] = "Low Risk"

        # Save transaction in database
        Transaction.objects.create(
            user=request.user,
            amount=transaction["amount"],
            hour=transaction["hour"],
            distance_km=transaction["distance_km"],
            device_score=transaction["device_score"],
            international=bool(
                transaction["international"]
            ),
            merchant_risk=transaction["merchant_risk"],

            high_amount=(
                transaction["amount"] > 5000
            ),
            night_transaction=(
                transaction["hour"] <= 5
                or transaction["hour"] >= 23
            ),
            unusual_distance=(
                transaction["distance_km"] > 20
            ),
            low_device_trust=(
                transaction["device_score"] < 0.40
            ),
            high_merchant_risk=(
                transaction["merchant_risk"] > 0.50
            ),

            fraud_probability=result[
                "fraud_probability"
            ],
            prediction=result["prediction"]
        )

    return render(
        request,
        "fraud_app/home.html",
        {"result": result}
    )
@login_required
def transaction_history(request):

    transactions = Transaction.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(
        request,
        "fraud_app/history.html",
        {"transactions": transactions}
    )
@login_required
def admin_history(request):

    if not request.user.is_staff:
        return redirect("home")

    transactions = Transaction.objects.all().order_by("-created_at")

    return render(
        request,
        "fraud_app/admin_history.html",
        {
            "transactions": transactions
        }
    )
@login_required
def dashboard(request):

    transactions = Transaction.objects.filter(
        user=request.user
    )

    total_transactions = transactions.count()

    fraud_transactions = transactions.filter(
        prediction="Fraudulent"
    ).count()

    legitimate_transactions = transactions.filter(
        prediction="Legitimate"
    ).count()

    if total_transactions > 0:
        average_probability = (
            sum(
                transaction.fraud_probability
                for transaction in transactions
            ) / total_transactions
        )
    else:
        average_probability = 0
    chart_data = [
    fraud_transactions,
    legitimate_transactions
 ]
    context = {
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_transactions,
        "legitimate_transactions": legitimate_transactions,
        "average_probability": round(
            average_probability, 2
        ),
        "chart_data": chart_data,
    }

    return render(
        request,
        "fraud_app/dashboard.html",
        context
    )
def user_logout(request):
    logout(request)

    return redirect("login")
def register(request):

    if request.method == "POST":

        full_name = request.POST.get("full_name")
        email = request.POST.get("email")
        username = request.POST.get("username")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if password != confirm_password:
            return render(
                request,
                "fraud_app/register.html",
                {"error": "Passwords do not match."}
            )

        if User.objects.filter(username=username).exists():
            return render(
                request,
                "fraud_app/register.html",
                {"error": "Username already exists."}
            )

        if User.objects.filter(email=email).exists():
            return render(
                request,
                "fraud_app/register.html",
                {"error": "Email already registered."}
            )

        User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=full_name
        )

        return render(
            request,
            "fraud_app/register.html",
            {"success": "Registration successful. You can now login."}
        )

    return render(
        request,
        "fraud_app/register.html"
    )
def user_login(request):

    if request.method == "POST":

        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:
            login(request, user)

            return redirect("home")

        return render(
            request,
            "fraud_app/login.html",
            {"error": "Invalid username or password."}
        )

    return render(
        request,
        "fraud_app/login.html"
    )
@login_required
def admin_dashboard(request):

    if not request.user.is_staff:
        return redirect("home")

    transactions = Transaction.objects.all()
    recent_transactions = transactions.order_by("-created_at")[:20]
    total_transactions = transactions.count()

    fraud_transactions = transactions.filter(
        prediction="Fraudulent"
    ).count()

    legitimate_transactions = transactions.filter(
        prediction="Legitimate"
    ).count()

    total_users = User.objects.count()

    if total_transactions > 0:
        average_probability = (
            sum(
                transaction.fraud_probability
                for transaction in transactions
            ) / total_transactions
        )
    else:
        average_probability = 0

    context = {
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_transactions,
        "legitimate_transactions": legitimate_transactions,
        "total_users": total_users,
        "average_probability": round(
            average_probability, 2
        ),
        "recent_transactions": recent_transactions,
    }

    return render(
        request,
        "fraud_app/admin_dashboard.html",
        context
    )
@login_required
@login_required
def csv_upload(request):

    if not request.user.is_staff:
        return redirect("home")

    result = None

    if request.method == "POST":

        csv_file = request.FILES.get("csv_file")

        if csv_file is None:
            return render(
                request,
                "fraud_app/csv_upload.html",
                {"error": "Please select a CSV file."}
            )

        try:

            df = pd.read_csv(csv_file)

            required_columns = [
                "amount",
                "hour",
                "distance_km",
                "device_score",
                "international",
                "merchant_risk"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                return render(
                    request,
                    "fraud_app/csv_upload.html",
                    {
                        "error": (
                            "Missing columns: "
                            + ", ".join(missing_columns)
                        )
                    }
                )

            fraud_count = 0
            legitimate_count = 0
            prediction_results = []

            for _, row in df.iterrows():

                transaction = {
                    "amount": float(row["amount"]),
                    "hour": int(row["hour"]),
                    "distance_km": float(row["distance_km"]),
                    "device_score": float(row["device_score"]),
                    "international": int(row["international"]),
                    "merchant_risk": float(row["merchant_risk"])
                }

                prediction_result = predict_fraud(transaction)

                prediction_results.append({
                    "amount": transaction["amount"],
                    "fraud_probability": prediction_result[
                        "fraud_probability"
                    ],
                    "prediction": prediction_result["prediction"]
                })

                if prediction_result["prediction"] == "Fraudulent":
                    fraud_count += 1
                else:
                    legitimate_count += 1

                Transaction.objects.create(
                    user=request.user,
                    amount=transaction["amount"],
                    hour=transaction["hour"],
                    distance_km=transaction["distance_km"],
                    device_score=transaction["device_score"],
                    international=bool(
                        transaction["international"]
                    ),
                    merchant_risk=transaction["merchant_risk"],
                    high_amount=(
                        transaction["amount"] > 5000
                    ),
                    night_transaction=(
                        transaction["hour"] <= 5
                        or transaction["hour"] >= 23
                    ),
                    unusual_distance=(
                        transaction["distance_km"] > 20
                    ),
                    low_device_trust=(
                        transaction["device_score"] < 0.40
                    ),
                    high_merchant_risk=(
                        transaction["merchant_risk"] > 0.50
                    ),
                    fraud_probability=(
                        prediction_result["fraud_probability"]
                    ),
                    prediction=(
                        prediction_result["prediction"]
                    )
                )

            result = {
                "total": len(df),
                "fraud": fraud_count,
                "legitimate": legitimate_count,
                "predictions": prediction_results
            }

        except Exception as e:

            return render(
                request,
                "fraud_app/csv_upload.html",
                {
                    "error": (
                        "Unable to process CSV file: "
                        + str(e)
                    )
                }
            )

    return render(
        request,
        "fraud_app/csv_upload.html",
        {"result": result}
    )