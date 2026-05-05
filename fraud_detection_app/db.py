from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "fraud_app.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            amount REAL NOT NULL,
            transaction_type TEXT NOT NULL,
            location TEXT NOT NULL,
            transaction_mode TEXT NOT NULL,
            device_type TEXT NOT NULL,
            merchant_category TEXT NOT NULL,
            age_group TEXT NOT NULL,
            hour INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            confidence REAL NOT NULL,
            prediction TEXT NOT NULL,
            model_used TEXT NOT NULL,
            user_avg_amount REAL NOT NULL,
            num_tx_24h INTEGER NOT NULL,
            is_new_location INTEGER NOT NULL,
            distance_from_last_km REAL NOT NULL,
            time_since_last_min REAL NOT NULL,
            geovelocity_kmph REAL NOT NULL,
            reasons_json TEXT NOT NULL,
            feature_json TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()


def fetch_user_by_username(username: str):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user


def create_user(username: str, password_hash: str, created_at: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, password_hash, created_at),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def ensure_demo_user(password_hash: str, created_at: str) -> None:
    if fetch_user_by_username("demo") is None:
        create_user("demo", password_hash, created_at)


def insert_transaction(values: dict[str, Any]) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO transactions (
            user_id, created_at, amount, transaction_type, location, transaction_mode,
            device_type, merchant_category, age_group, hour, risk_score, confidence,
            prediction, model_used, user_avg_amount, num_tx_24h, is_new_location,
            distance_from_last_km, time_since_last_min, geovelocity_kmph,
            reasons_json, feature_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            values["user_id"],
            values["created_at"],
            values["amount"],
            values["transaction_type"],
            values["location"],
            values["transaction_mode"],
            values["device_type"],
            values["merchant_category"],
            values["age_group"],
            values["hour"],
            values["risk_score"],
            values["confidence"],
            values["prediction"],
            values["model_used"],
            values["user_avg_amount"],
            values["num_tx_24h"],
            values["is_new_location"],
            values["distance_from_last_km"],
            values["time_since_last_min"],
            values["geovelocity_kmph"],
            values["reasons_json"],
            values["feature_json"],
        ),
    )
    conn.commit()
    conn.close()


def fetch_user_transactions(user_id: int, limit: int | None = None):
    conn = get_connection()
    query = "SELECT * FROM transactions WHERE user_id = ? ORDER BY datetime(created_at) DESC"
    params: list[Any] = [user_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def transaction_count_for_user(user_id: int) -> int:
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) AS count FROM transactions WHERE user_id = ?", (user_id,)
    ).fetchone()["count"]
    conn.close()
    return count
