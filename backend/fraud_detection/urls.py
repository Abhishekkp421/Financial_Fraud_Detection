from django.contrib import admin
from django.urls import path
from fraud_app.views import home, transaction_history, register,user_login,user_logout,dashboard,admin_dashboard ,csv_upload,admin_history

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home, name="home"),
    path("history/", transaction_history, name="history"),
    path("register/", register, name="register"),
    path("login/", user_login, name="login"),
    path("logout/", user_logout, name="logout"),
    path("dashboard/", dashboard, name="dashboard"),
    path("admin-dashboard/", admin_dashboard, name="admin_dashboard"),
    path("csv-upload/", csv_upload, name="csv_upload"),
    path("admin-history/", admin_history, name="admin_history"),
]