from __future__ import annotations

import json
import math
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

CITY_COORDS = {
    "Hyderabad": (17.3850, 78.4867),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Dubai": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198),
    "London": (51.5072, -0.1276),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093),
}

TRANSACTION_TYPES = ["Payment", "Transfer", "Withdrawal", "Online Purchase"]
TRANSACTION_MODES = ["Mobile", "Web", "ATM", "POS"]
DEVICE_TYPES = ["Android", "iOS", "Desktop"]
MERCHANT_CATEGORIES = ["Shopping", "Food", "Travel", "Bills", "Others"]
AGE_GROUPS = ["Young", "Adult", "Senior"]

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "user_avg_amount",
    "amount_ratio",
    "num_tx_24h",
    "is_new_location",
    "distance_from_last_km",
    "time_since_last_min",
    "geovelocity_kmph",
    "is_night",
]
CATEGORICAL_FEATURES = [
    "transaction_type",
    "location",
    "transaction_mode",
    "device_type",
    "merchant_category",
    "age_group",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def haversine_km(city_a: str, city_b: str) -> float:
    lat1, lon1 = CITY_COORDS[city_a]
    lat2, lon2 = CITY_COORDS[city_b]
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sample_amount(transaction_type: str, merchant_category: str) -> float:
    base = {
        "Payment": 1200,
        "Transfer": 4500,
        "Withdrawal": 2500,
        "Online Purchase": 3200,
    }[transaction_type]
    category_boost = {
        "Shopping": 1.2,
        "Food": 0.6,
        "Travel": 2.2,
        "Bills": 1.5,
        "Others": 1.0,
    }[merchant_category]
    noise = np.random.lognormal(mean=0.0, sigma=0.7)
    return round(max(50, base * category_boost * noise), 2)


def generate_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    city_list = list(CITY_COORDS)
    for _ in range(n_samples):
        transaction_type = random.choice(TRANSACTION_TYPES)
        merchant_category = random.choice(MERCHANT_CATEGORIES)
        transaction_mode = random.choice(TRANSACTION_MODES)
        device_type = random.choice(DEVICE_TYPES)
        age_group = random.choices(AGE_GROUPS, weights=[0.25, 0.6, 0.15], k=1)[0]
        location = random.choice(city_list)
        previous_location = random.choice(city_list)
        hour = random.randint(0, 23)
        is_night = int(hour <= 5 or hour >= 23)

        amount = sample_amount(transaction_type, merchant_category)
        user_avg_amount = max(200, np.random.lognormal(mean=7.3, sigma=0.55))
        amount_ratio = round(amount / max(user_avg_amount, 1), 3)
        num_tx_24h = min(int(np.random.poisson(lam=2.6)) + 1, 14)

        is_new_location = int(location != previous_location and random.random() < 0.7)
        if not is_new_location:
            previous_location = location
        distance = haversine_km(location, previous_location) if location != previous_location else 0.0
        time_since_last_min = float(np.clip(np.random.lognormal(mean=3.2, sigma=1.0), 2, 2500))
        geovelocity = distance / max(time_since_last_min / 60.0, 1e-3)

        rule_score = -7.0
        rule_score += 1.8 if amount > 8000 else 0.0
        rule_score += 1.6 if amount_ratio > 3.5 else 0.0
        rule_score += 1.0 if amount_ratio > 2.0 else 0.0
        rule_score += 1.3 if num_tx_24h >= 7 else 0.0
        rule_score += 1.6 if is_new_location else 0.0
        rule_score += 1.7 if distance > 700 else 0.0
        rule_score += 2.0 if geovelocity > 900 else 0.0
        rule_score += 1.1 if is_night and transaction_type in {"Online Purchase", "Transfer"} else 0.0
        rule_score += 0.8 if transaction_mode == "Web" and device_type == "Desktop" and is_night else 0.0
        rule_score += 0.9 if merchant_category == "Travel" and amount > user_avg_amount * 2 else 0.0
        rule_score += 0.4 if age_group == "Senior" and transaction_mode == "Web" else 0.0
        rule_score += float(np.random.normal(0, 0.5))
        fraud_probability = np.clip(sigmoid(rule_score), 0.01, 0.99)
        label = int(np.random.random() < fraud_probability)

        rows.append(
            {
                "amount": amount,
                "hour": hour,
                "user_avg_amount": round(float(user_avg_amount), 2),
                "amount_ratio": amount_ratio,
                "num_tx_24h": num_tx_24h,
                "is_new_location": is_new_location,
                "distance_from_last_km": round(distance, 2),
                "time_since_last_min": round(time_since_last_min, 2),
                "geovelocity_kmph": round(geovelocity, 2),
                "is_night": is_night,
                "transaction_type": transaction_type,
                "location": location,
                "transaction_mode": transaction_mode,
                "device_type": device_type,
                "merchant_category": merchant_category,
                "age_group": age_group,
                "fraud_label": label,
            }
        )
    return pd.DataFrame(rows)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test, preprocessor):
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    return {
        "name": name,
        "model": model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "positive_rate": float(y_prob.mean()),
    }


def plot_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Safe", "Fraud"],
        yticklabels=["Safe", "Fraud"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = generate_dataset()
    X = df[FEATURE_COLUMNS].copy()
    y = df["fraud_label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    rf_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model = XGBClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )

    rf_metrics = evaluate_model(
        "Random Forest",
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )
    xgb_metrics = evaluate_model(
        "XGBoost",
        xgb_model,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

    models = {"Random Forest": rf_metrics, "XGBoost": xgb_metrics}
    best_name = max(models, key=lambda name: (models[name]["f1"], models[name]["recall"]))
    best_metrics = models[best_name]

    feature_names = preprocessor.get_feature_names_out().tolist()
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "rf_model": rf_metrics["model"],
            "xgb_model": xgb_metrics["model"],
            "best_model_name": best_name,
            "feature_names": feature_names,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "feature_columns": FEATURE_COLUMNS,
            "city_coords": CITY_COORDS,
        },
        MODELS_DIR / "fraud_bundle.joblib",
    )

    output_metrics = {
        "dataset_size": int(len(df)),
        "fraud_rate": round(float(df["fraud_label"].mean()), 4),
        "best_model": best_name,
        "models": {
            name: {
                key: value
                for key, value in metrics.items()
                if key not in {"model"}
            }
            for name, metrics in models.items()
        },
    }

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=2)

    plot_confusion_matrix(best_metrics["confusion_matrix"], STATIC_DIR / "confusion_matrix_best.png")
    plot_confusion_matrix(rf_metrics["confusion_matrix"], STATIC_DIR / "confusion_matrix_rf.png")
    plot_confusion_matrix(xgb_metrics["confusion_matrix"], STATIC_DIR / "confusion_matrix_xgb.png")

    print(json.dumps(output_metrics, indent=2))


if __name__ == "__main__":
    main()
