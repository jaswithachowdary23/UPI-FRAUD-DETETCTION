from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
BUNDLE_PATH = MODELS_DIR / "fraud_bundle.joblib"

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "hour": "Transaction hour",
    "user_avg_amount": "Usual spending baseline",
    "amount_ratio": "Amount vs usual spending",
    "num_tx_24h": "Transactions in last 24 hours",
    "is_new_location": "New location check",
    "distance_from_last_km": "Distance from previous location",
    "time_since_last_min": "Time since previous transaction",
    "geovelocity_kmph": "Travel velocity check",
    "is_night": "Odd transaction time",
    "transaction_type": "Transaction type",
    "location": "Current location",
    "transaction_mode": "Channel used",
    "device_type": "Device type",
    "merchant_category": "Merchant category",
    "age_group": "User age group",
}


@dataclass
class BehaviorProfile:
    user_avg_amount: float
    num_tx_24h: int
    common_locations: list[str]
    active_hours: list[int]
    last_location: str | None
    last_timestamp: datetime | None
    time_since_last_min: float
    distance_from_last_km: float
    geovelocity_kmph: float
    is_new_location: int


class FraudModelService:
    def __init__(self) -> None:
        if not BUNDLE_PATH.exists() or not METRICS_PATH.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run train_model.py before starting the app."
            )
        bundle = joblib.load(BUNDLE_PATH)
        self.preprocessor = bundle["preprocessor"]
        self.models = {
            "Random Forest": bundle["rf_model"],
            "XGBoost": bundle["xgb_model"],
        }
        self.best_model_name = bundle["best_model_name"]
        self.best_model = self.models[self.best_model_name]
        self.feature_names = bundle["feature_names"]
        self.numeric_features = bundle["numeric_features"]
        self.categorical_features = bundle["categorical_features"]
        self.feature_columns = bundle["feature_columns"]
        self.city_coords = bundle["city_coords"]
        self.explainer = shap.TreeExplainer(self.best_model)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

    def haversine_km(self, city_a: str, city_b: str) -> float:
        if city_a not in self.city_coords or city_b not in self.city_coords:
            return 0.0
        lat1, lon1 = self.city_coords[city_a]
        lat2, lon2 = self.city_coords[city_b]
        r = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def build_profile(
        self,
        current_location: str,
        current_timestamp: datetime,
        transaction_history: list[dict[str, Any]],
    ) -> BehaviorProfile:
        if not transaction_history:
            return BehaviorProfile(
                user_avg_amount=1800.0,
                num_tx_24h=1,
                common_locations=[current_location],
                active_hours=[current_timestamp.hour],
                last_location=None,
                last_timestamp=None,
                time_since_last_min=720.0,
                distance_from_last_km=0.0,
                geovelocity_kmph=0.0,
                is_new_location=0,
            )

        amounts = [float(row["amount"]) for row in transaction_history]
        user_avg_amount = round(sum(amounts) / len(amounts), 2)
        recent = [
            row
            for row in transaction_history
            if (current_timestamp - datetime.fromisoformat(row["created_at"]))
            .total_seconds()
            <= 24 * 3600
        ]
        num_tx_24h = max(1, len(recent) + 1)

        location_counts: dict[str, int] = {}
        active_hour_counts: dict[int, int] = {}
        for row in transaction_history:
            location_counts[row["location"]] = location_counts.get(row["location"], 0) + 1
            active_hour_counts[int(row["hour"])] = active_hour_counts.get(int(row["hour"]), 0) + 1
        common_locations = sorted(location_counts, key=location_counts.get, reverse=True)[:3]
        active_hours = sorted(active_hour_counts, key=active_hour_counts.get, reverse=True)[:4]

        last_row = max(transaction_history, key=lambda item: item["created_at"])
        last_timestamp = datetime.fromisoformat(last_row["created_at"])
        last_location = last_row["location"]
        time_since_last_min = max((current_timestamp - last_timestamp).total_seconds() / 60.0, 1.0)
        distance = self.haversine_km(current_location, last_location)
        geovelocity = distance / max(time_since_last_min / 60.0, 1e-3)
        is_new_location = int(current_location not in common_locations)

        return BehaviorProfile(
            user_avg_amount=user_avg_amount,
            num_tx_24h=num_tx_24h,
            common_locations=common_locations,
            active_hours=active_hours,
            last_location=last_location,
            last_timestamp=last_timestamp,
            time_since_last_min=round(time_since_last_min, 2),
            distance_from_last_km=round(distance, 2),
            geovelocity_kmph=round(geovelocity, 2),
            is_new_location=is_new_location,
        )

    def prepare_features(
        self,
        raw_input: dict[str, Any],
        profile: BehaviorProfile,
    ) -> pd.DataFrame:
        amount = float(raw_input["amount"])
        timestamp = datetime.fromisoformat(raw_input["timestamp"])
        row = {
            "amount": amount,
            "hour": timestamp.hour,
            "user_avg_amount": profile.user_avg_amount,
            "amount_ratio": round(amount / max(profile.user_avg_amount, 1.0), 3),
            "num_tx_24h": profile.num_tx_24h,
            "is_new_location": profile.is_new_location,
            "distance_from_last_km": profile.distance_from_last_km,
            "time_since_last_min": profile.time_since_last_min,
            "geovelocity_kmph": profile.geovelocity_kmph,
            "is_night": int(timestamp.hour <= 5 or timestamp.hour >= 23),
            "transaction_type": raw_input["transaction_type"],
            "location": raw_input["location"],
            "transaction_mode": raw_input["transaction_mode"],
            "device_type": raw_input["device_type"],
            "merchant_category": raw_input["merchant_category"],
            "age_group": raw_input["age_group"],
        }
        return pd.DataFrame([row], columns=self.feature_columns)

    def _map_feature_name(self, transformed_name: str) -> str:
        if transformed_name.startswith("num__"):
            return transformed_name.replace("num__", "", 1)
        for feature in self.categorical_features:
            prefix = f"cat__{feature}_"
            if transformed_name.startswith(prefix):
                return feature
        return transformed_name

    def _reason_text(self, feature: str, row: pd.Series, positive: bool = True) -> str:
        if positive:
            if feature == "amount_ratio":
                return f"Amount is {row['amount_ratio']:.1f}× higher than the user's usual spending."
            if feature == "is_new_location":
                return "Transaction is coming from a new location for this user."
            if feature == "geovelocity_kmph":
                return f"Travel velocity looks unrealistic at about {row['geovelocity_kmph']:.0f} km/h."
            if feature == "distance_from_last_km":
                return f"Current transaction is {row['distance_from_last_km']:.0f} km away from the previous location."
            if feature == "num_tx_24h":
                return f"High activity detected with {int(row['num_tx_24h'])} transactions in the last 24 hours."
            if feature == "is_night":
                return "Transaction happened during unusual late-night hours."
            if feature == "merchant_category":
                return f"Merchant category '{row['merchant_category']}' is contributing to the risk pattern."
            if feature == "transaction_type":
                return f"Transaction type '{row['transaction_type']}' matches a higher-risk pattern."
            if feature == "transaction_mode":
                return f"Channel '{row['transaction_mode']}' is contributing to the risk score."
            if feature == "device_type":
                return f"Device '{row['device_type']}' differs from a safer historical pattern."
            if feature == "location":
                return f"Location '{row['location']}' is influencing the fraud score."
            return f"{FEATURE_LABELS.get(feature, feature)} is contributing to the fraud score."

        if feature == "amount_ratio":
            return "Amount is still close to the user's normal spending pattern."
        if feature == "is_new_location":
            return "Location behavior does not strongly deviate from the user's baseline."
        if feature == "geovelocity_kmph":
            return "No impossible travel pattern is being detected here."
        if feature == "distance_from_last_km":
            return "Distance from the previous transaction does not strongly raise fraud risk."
        if feature == "num_tx_24h":
            return "Recent transaction count remains within an acceptable range."
        if feature == "is_night":
            return "Transaction timing is not a major risk contributor in this case."
        return f"{FEATURE_LABELS.get(feature, feature)} is aligned with a safer transaction pattern."

    def explain_prediction(self, features_df: pd.DataFrame, prefer_positive: bool = True) -> list[dict[str, Any]]:
        transformed = self.preprocessor.transform(features_df)
        shap_values = self.explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_row = shap_values[0, :, -1]
        elif shap_values.ndim == 2:
            shap_row = shap_values[0]
        else:
            shap_row = shap_values.reshape(-1)
        contributions: dict[str, float] = {}
        for transformed_name, value in zip(self.feature_names, shap_row):
            original_name = self._map_feature_name(transformed_name)
            contributions[original_name] = contributions.get(original_name, 0.0) + float(value)

        if prefer_positive:
            filtered = [(feature, value) for feature, value in contributions.items() if value > 0]
            sorted_contribs = sorted(filtered, key=lambda item: item[1], reverse=True) or sorted(
                contributions.items(), key=lambda item: abs(item[1]), reverse=True
            )
        else:
            filtered = [(feature, value) for feature, value in contributions.items() if value < 0]
            sorted_contribs = sorted(filtered, key=lambda item: item[1]) or sorted(
                contributions.items(), key=lambda item: abs(item[1]), reverse=True
            )
        top = []
        row = features_df.iloc[0]
        for feature, value in sorted_contribs[:5]:
            top.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                    "impact": round(value, 4),
                    "reason": self._reason_text(feature, row, positive=prefer_positive),
                }
            )
        return top

    def predict(self, raw_input: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        current_timestamp = datetime.fromisoformat(raw_input["timestamp"])
        profile = self.build_profile(raw_input["location"], current_timestamp, history)
        features_df = self.prepare_features(raw_input, profile)
        transformed = self.preprocessor.transform(features_df)

        model_scores = {}
        for model_name, model in self.models.items():
            fraud_prob = float(model.predict_proba(transformed)[0][1])
            model_scores[model_name] = round(fraud_prob * 100, 2)

        best_prob = float(self.best_model.predict_proba(transformed)[0][1])
        risk_score = round(best_prob * 100, 2)
        confidence = round(max(best_prob, 1 - best_prob) * 100, 2)
        prediction = "Fraud" if best_prob >= 0.5 else "Not Fraud"
        alert = risk_score >= 75.0
        explanations = self.explain_prediction(features_df, prefer_positive=best_prob >= 0.5)

        geo_flag = (
            profile.last_location is not None
            and profile.distance_from_last_km > 500
            and profile.time_since_last_min < 120
        )
        heuristic_notes = []
        if features_df.iloc[0]["amount_ratio"] > 3.0:
            heuristic_notes.append("Unusual transaction amount")
        if profile.is_new_location:
            heuristic_notes.append("New location detected")
        if features_df.iloc[0]["is_night"]:
            heuristic_notes.append("Odd transaction time")
        if geo_flag:
            heuristic_notes.append(
                f"Geo-velocity mismatch: {profile.last_location} → {raw_input['location']} in {profile.time_since_last_min:.0f} mins"
            )
        if features_df.iloc[0]["num_tx_24h"] >= 7:
            heuristic_notes.append("Spike in transaction frequency")
        if not heuristic_notes:
            heuristic_notes.append("Behavior matches normal user spending pattern")

        return {
            "prediction": prediction,
            "risk_score": risk_score,
            "confidence": confidence,
            "alert": alert,
            "model_used": self.best_model_name,
            "all_model_scores": model_scores,
            "threshold": 50,
            "alert_threshold": 75,
            "profile": {
                "user_avg_amount": profile.user_avg_amount,
                "num_tx_24h": profile.num_tx_24h,
                "common_locations": profile.common_locations,
                "active_hours": profile.active_hours,
                "last_location": profile.last_location,
                "time_since_last_min": profile.time_since_last_min,
                "distance_from_last_km": profile.distance_from_last_km,
                "geovelocity_kmph": profile.geovelocity_kmph,
            },
            "explanations": explanations,
            "heuristic_notes": heuristic_notes,
            "engineered_features": features_df.iloc[0].to_dict(),
        }
