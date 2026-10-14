from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Dict, Any, List


DB_PATH = Path("data/predictions.db")


def init_db(db_path: Path = DB_PATH) -> None:
    """
    ایجاد جدول لاگ پیش‌بینی‌ها در پایگاه داده SQLite در صورت عدم وجود.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                is_anomaly INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                threshold REAL NOT NULL,
                status TEXT NOT NULL,
                top_contributing_sensor TEXT NOT NULL,
                sensor_deviation_zscore REAL NOT NULL,
                raw_sensor_payload TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(
    sensor_payload: Dict[str, float],
    prediction_result: Dict[str, Any],
    db_path: Path = DB_PATH,
) -> int:
    """
    ثبت یک رکورد پیش‌بینی جدید به همراه داده‌های سنسور ورودی در دیتابیس.
    شناسه رکورد ثبت‌شده (record_id) را برمی‌گرداند.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_logs (
                timestamp, is_anomaly, anomaly_score, threshold,
                status, top_contributing_sensor, sensor_deviation_zscore, raw_sensor_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                int(prediction_result["is_anomaly"]),
                prediction_result["anomaly_score"],
                prediction_result["threshold"],
                prediction_result["status"],
                prediction_result["top_contributing_sensor"],
                prediction_result["sensor_deviation_zscore"],
                json.dumps(sensor_payload),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_recent_predictions(limit: int = 20, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """
    واکشی آخرین پیش‌بینی‌های ثبت‌شده برای نمایش در داشبورد یا مانیتورینگ.
    """
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["is_anomaly"] = bool(item["is_anomaly"])
        item["raw_sensor_payload"] = json.loads(item["raw_sensor_payload"])
        results.append(item)
    return results