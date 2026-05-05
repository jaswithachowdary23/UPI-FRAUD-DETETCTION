from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from db import (
    create_user,
    ensure_demo_user,
    fetch_user_by_username,
    fetch_user_transactions,
    get_connection,
    init_db,
    insert_transaction,
    transaction_count_for_user,
)
from model_service import FraudModelService
from train_model import main as train_main

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models" / "fraud_bundle.joblib"

if not MODELS_PATH.exists():
    train_main()

init_db()
ensure_demo_user(generate_password_hash("demo123"), datetime.utcnow().isoformat())
model_service = FraudModelService()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "fraud-detection-demo-secret")


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    g.user = None
    if user_id:
        conn = get_connection()
        g.user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()


@app.context_processor
def inject_globals():
    return {
        "available_locations": list(model_service.city_coords.keys()),
        "transaction_types": ["Payment", "Transfer", "Withdrawal", "Online Purchase"],
        "transaction_modes": ["Mobile", "Web", "ATM", "POS"],
        "device_types": ["Android", "iOS", "Desktop"],
        "merchant_categories": ["Shopping", "Food", "Travel", "Bills", "Others"],
        "age_groups": ["Young", "Adult", "Senior"],
        "model_service": model_service,
    }



def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view



def summarize_transactions(rows):
    total = len(rows)
    fraud_count = sum(1 for row in rows if row["prediction"] == "Fraud")
    safe_count = total - fraud_count
    fraud_percentage = round((fraud_count / total) * 100, 2) if total else 0.0
    avg_risk = round(sum(float(row["risk_score"]) for row in rows) / total, 2) if total else 0.0

    by_location = Counter(row["location"] for row in rows)
    by_type = Counter(row["transaction_type"] for row in rows)
    by_prediction = {"Fraud": fraud_count, "Safe": safe_count}
    daily_counts = Counter(row["created_at"][:10] for row in rows)

    return {
        "total": total,
        "fraud_count": fraud_count,
        "safe_count": safe_count,
        "fraud_percentage": fraud_percentage,
        "avg_risk": avg_risk,
        "location_labels": list(by_location.keys()),
        "location_values": list(by_location.values()),
        "type_labels": list(by_type.keys()),
        "type_values": list(by_type.values()),
        "prediction_labels": list(by_prediction.keys()),
        "prediction_values": list(by_prediction.values()),
        "daily_labels": sorted(daily_counts.keys()),
        "daily_values": [daily_counts[key] for key in sorted(daily_counts.keys())],
    }



def build_behavior_snapshot(rows):
    if not rows:
        return {
            "usual_spending_range": "₹0 - ₹0",
            "common_locations": [],
            "active_hours": "No history yet",
            "last_seen": "No transactions yet",
        }

    amounts = [float(row["amount"]) for row in rows]
    lower = min(amounts)
    upper = max(amounts)
    common_locations = [item for item, _ in Counter(row["location"] for row in rows).most_common(3)]
    hour_counts = Counter(int(row["hour"]) for row in rows)
    busiest_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:3]
    last_seen = max(row["created_at"] for row in rows)
    return {
        "usual_spending_range": f"₹{lower:,.0f} - ₹{upper:,.0f}",
        "common_locations": common_locations,
        "active_hours": ", ".join(f"{hour:02d}:00" for hour in sorted(busiest_hours)),
        "last_seen": last_seen.replace("T", " ")[:19],
    }



def seed_demo_transactions_if_needed() -> None:
    demo_user = fetch_user_by_username("demo")
    if demo_user is None or transaction_count_for_user(demo_user["id"]) > 0:
        return

    examples = [
        {
            "amount": 1250,
            "transaction_type": "Payment",
            "location": "Hyderabad",
            "transaction_mode": "Mobile",
            "device_type": "Android",
            "merchant_category": "Bills",
            "age_group": "Adult",
            "timestamp": "2026-04-22T10:15:00",
        },
        {
            "amount": 18450,
            "transaction_type": "Online Purchase",
            "location": "New York",
            "transaction_mode": "Web",
            "device_type": "Desktop",
            "merchant_category": "Travel",
            "age_group": "Adult",
            "timestamp": "2026-04-22T23:40:00",
        },
        {
            "amount": 2600,
            "transaction_type": "Withdrawal",
            "location": "Hyderabad",
            "transaction_mode": "ATM",
            "device_type": "Android",
            "merchant_category": "Others",
            "age_group": "Adult",
            "timestamp": "2026-04-23T09:05:00",
        },
    ]
    history = []
    for payload in examples:
        result = model_service.predict(payload, history)
        engineered = result["engineered_features"]
        insert_transaction(
            {
                "user_id": demo_user["id"],
                "created_at": payload["timestamp"],
                "amount": payload["amount"],
                "transaction_type": payload["transaction_type"],
                "location": payload["location"],
                "transaction_mode": payload["transaction_mode"],
                "device_type": payload["device_type"],
                "merchant_category": payload["merchant_category"],
                "age_group": payload["age_group"],
                "hour": engineered["hour"],
                "risk_score": result["risk_score"],
                "confidence": result["confidence"],
                "prediction": result["prediction"],
                "model_used": result["model_used"],
                "user_avg_amount": engineered["user_avg_amount"],
                "num_tx_24h": engineered["num_tx_24h"],
                "is_new_location": engineered["is_new_location"],
                "distance_from_last_km": engineered["distance_from_last_km"],
                "time_since_last_min": engineered["time_since_last_min"],
                "geovelocity_kmph": engineered["geovelocity_kmph"],
                "reasons_json": json.dumps(result["explanations"]),
                "feature_json": json.dumps(engineered),
            }
        )
        history = [dict(row) for row in fetch_user_transactions(demo_user["id"])]


seed_demo_transactions_if_needed()


@app.route("/")
@login_required
def index():
    rows = [dict(row) for row in fetch_user_transactions(g.user["id"], limit=8)]
    summary = summarize_transactions(rows)
    behavior = build_behavior_snapshot(rows)
    return render_template("index.html", summary=summary, behavior=behavior, recent_rows=rows)


@app.route("/dashboard")
@login_required
def dashboard():
    rows = [dict(row) for row in fetch_user_transactions(g.user["id"])]
    summary = summarize_transactions(rows)
    behavior = build_behavior_snapshot(rows)
    return render_template("dashboard.html", summary=summary, behavior=behavior)


@app.route("/history")
@login_required
def history():
    rows = [dict(row) for row in fetch_user_transactions(g.user["id"])]
    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                **row,
                "reasons": json.loads(row["reasons_json"]),
            }
        )
    return render_template("history.html", rows=parsed_rows)


@app.route("/performance")
@login_required
def performance():
    metrics = model_service.metrics
    return render_template("performance.html", metrics=metrics)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = fetch_user_by_username(username)
        error = None
        if user is None or not check_password_hash(user["password_hash"], password):
            error = "Invalid username or password."
        if error:
            flash(error, "danger")
        else:
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        error = None
        if not username:
            error = "Username is required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif fetch_user_by_username(username) is not None:
            error = "Username already exists."
        if error:
            flash(error, "danger")
        else:
            user_id = create_user(username, generate_password_hash(password), datetime.utcnow().isoformat())
            session.clear()
            session["user_id"] = user_id
            flash("Account created successfully.", "success")
            return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    required_fields = [
        "amount",
        "transaction_type",
        "location",
        "transaction_mode",
        "device_type",
        "merchant_category",
        "age_group",
    ]
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        amount = float(payload.get("amount"))
    except ValueError:
        return jsonify({"error": "Amount must be numeric."}), 400

    timestamp = payload.get("timestamp") or datetime.utcnow().replace(microsecond=0).isoformat()
    raw_input = {
        "amount": amount,
        "transaction_type": payload["transaction_type"],
        "location": payload["location"],
        "transaction_mode": payload["transaction_mode"],
        "device_type": payload["device_type"],
        "merchant_category": payload["merchant_category"],
        "age_group": payload["age_group"],
        "timestamp": timestamp,
    }

    history_rows = [dict(row) for row in fetch_user_transactions(g.user["id"])]
    result = model_service.predict(raw_input, history_rows)
    engineered = result["engineered_features"]

    insert_transaction(
        {
            "user_id": g.user["id"],
            "created_at": timestamp,
            "amount": amount,
            "transaction_type": raw_input["transaction_type"],
            "location": raw_input["location"],
            "transaction_mode": raw_input["transaction_mode"],
            "device_type": raw_input["device_type"],
            "merchant_category": raw_input["merchant_category"],
            "age_group": raw_input["age_group"],
            "hour": engineered["hour"],
            "risk_score": result["risk_score"],
            "confidence": result["confidence"],
            "prediction": result["prediction"],
            "model_used": result["model_used"],
            "user_avg_amount": engineered["user_avg_amount"],
            "num_tx_24h": engineered["num_tx_24h"],
            "is_new_location": engineered["is_new_location"],
            "distance_from_last_km": engineered["distance_from_last_km"],
            "time_since_last_min": engineered["time_since_last_min"],
            "geovelocity_kmph": engineered["geovelocity_kmph"],
            "reasons_json": json.dumps(result["explanations"]),
            "feature_json": json.dumps(engineered),
        }
    )

    refreshed_rows = [dict(row) for row in fetch_user_transactions(g.user["id"], limit=8)]
    result["summary"] = summarize_transactions(refreshed_rows)
    result["recent_transactions"] = [
        {
            "created_at": row["created_at"],
            "amount": row["amount"],
            "prediction": row["prediction"],
            "risk_score": row["risk_score"],
        }
        for row in refreshed_rows[:5]
    ]
    return jsonify(result)


@app.route("/api/dashboard-stats")
@login_required
def dashboard_stats():
    rows = [dict(row) for row in fetch_user_transactions(g.user["id"])]
    return jsonify({"summary": summarize_transactions(rows), "behavior": build_behavior_snapshot(rows)})


if __name__ == "__main__":
    app.run(debug=True)
