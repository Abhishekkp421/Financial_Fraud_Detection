from django.db import models
from django.contrib.auth.models import User


class Transaction(models.Model):
    user = models.ForeignKey(
    User,
    on_delete=models.CASCADE,
    null=True,
    blank=True
    )
    amount = models.FloatField()

    hour = models.IntegerField()

    distance_km = models.FloatField()

    device_score = models.FloatField()

    international = models.BooleanField()

    merchant_risk = models.FloatField()

    high_amount = models.BooleanField(default=False)

    night_transaction = models.BooleanField(default=False)

    unusual_distance = models.BooleanField(default=False)

    low_device_trust = models.BooleanField(default=False)

    high_merchant_risk = models.BooleanField(default=False)

    fraud_probability = models.FloatField()

    prediction = models.CharField(
        max_length=20
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )


    def __str__(self):

        return (
            f"{self.prediction} - "
            f"{self.amount}"
        )