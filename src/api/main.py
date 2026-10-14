from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.api.database import init_db, log_prediction, fetch_recent_predictions
from src.api.schemas import AnomalyPredictionResponse, SensorReadingRequest
from src.models.predictor import AnomalyPredictor


PREDICTOR_PATH = Path("models/predictor.joblib")
predictor_instance: AnomalyPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    راه‌اندازی دیتابیس و بارگذاری مدل در حافظه هنگام شروع به کار سرور.
    """
    global predictor_instance
    init_db()
    if PREDICTOR_PATH.exists():
        predictor_instance = AnomalyPredictor.load(PREDICTOR_PATH)
        print(f"Predictor loaded successfully from {PREDICTOR_PATH}")
    else:
        print(f"WARNING: Predictor artifact not found at {PREDICTOR_PATH}. Run predictor.py first.")
    yield


app = FastAPI(
    title="Industrial Anomaly Detection Platform",
    description="Real-time multivariate sensor anomaly detection, root-cause isolation, and drift monitoring API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    بررسی سلامت سرویس و وضعیت بارگذاری مدل در حافظه.
    """
    return {
        "status": "healthy",
        "model_loaded": predictor_instance is not None,
    }


@app.post("/predict", response_model=AnomalyPredictionResponse)
def predict_anomaly(reading: SensorReadingRequest):
    """
    دریافت لحظه‌ای مقادیر ۸ سنسور، اجرای استنتاج، ذخیره در دیتابیس و بازگرداندن نتیجه.
    """
    if predictor_instance is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    payload = reading.model_dump(by_alias=True)
    result = predictor_instance.predict_single(payload)
    record_id = log_prediction(payload, result)
    result["record_id"] = record_id
    return result


@app.get("/history", response_model=List[Dict[str, Any]])
def get_prediction_history(limit: int = Query(default=30, ge=1, le=200)):
    """
    دریافت تاریخچه آخرین پیش‌بینی‌ها و هشدارهای ثبت‌شده در پایگاه داده.
    """
    return fetch_recent_predictions(limit=limit)


@app.get("/drift")
def check_data_drift(window_size: int = Query(default=50, ge=5, le=500)):
    """
    بررسی پدیده Data Drift با مقایسه میانگین پنجره اخیر داده‌های ورودی نسبت به توزیع زمان آموزش.
    اگر میانگین یک سنسور بیش از 2 برابر انحراف معیار پایه جابه‌جا شده باشد، هشدار Drift صادر می‌شود.
    """
    if predictor_instance is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    recent_logs = fetch_recent_predictions(limit=window_size)
    if len(recent_logs) < 5:
        return {"status": "INSUFFICIENT_DATA", "samples_analyzed": len(recent_logs), "drifted_sensors": {}}

    sensor_records = [row["raw_sensor_payload"] for row in recent_logs]
    drifted_sensors = {}

    for sensor in predictor_instance.model.feature_names:
        recent_mean = float(np.mean([r[sensor] for r in sensor_records]))
        train_mean = float(predictor_instance.sensor_means[sensor])
        train_std = float(predictor_instance.sensor_stds[sensor])
        shift_z = abs(recent_mean - train_mean) / train_std

        if shift_z > 2.0:
            drifted_sensors[sensor] = round(shift_z, 2)

    return {
        "status": "DRIFT_DETECTED" if drifted_sensors else "STABLE",
        "samples_analyzed": len(recent_logs),
        "drifted_sensors": drifted_sensors,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def live_dashboard():
    """
    داشبورد وب بلادرنگ برای نمایش نمودار امتیاز ناهنجاری، خط آستانه و جدول هشدارها.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Industrial Anomaly Detection Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Segoe UI, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 15px; }
            .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
            .card { background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
            .card h3 { margin: 0 0 8px 0; font-size: 13px; color: #94a3b8; text-transform: uppercase; }
            .card .val { font-size: 24px; font-weight: bold; }
            .chart-box { background: #1e293b; padding: 20px; border-radius: 8px; margin-top: 20px; border: 1px solid #334155; height: 320px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }
            th { background: #0f172a; color: #94a3b8; }
            .badge-crit { background: #ef4444; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
            .badge-norm { background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Industrial Sensor Anomaly & Predictive Maintenance Monitor</h2>
            <span id="drift-status" style="color: #38bdf8;">Checking Drift...</span>
        </div>
        <div class="cards">
            <div class="card"><h3>System Status</h3><div id="sys-status" class="val">-</div></div>
            <div class="card"><h3>Latest Anomaly Score</h3><div id="latest-score" class="val">-</div></div>
            <div class="card"><h3>Top Suspect Sensor</h3><div id="top-sensor" class="val">-</div></div>
            <div class="card"><h3>Recent Alerts (Last 30)</h3><div id="alert-count" class="val">-</div></div>
        </div>
        <div class="chart-box">
            <canvas id="scoreChart"></canvas>
        </div>
        <table>
            <thead>
                <tr><th>ID</th><th>Timestamp (UTC)</th><th>Status</th><th>Score</th><th>Threshold</th><th>Top Contributing Sensor</th><th>Sensor Z-Score</th></tr>
            </thead>
            <tbody id="log-table"></tbody>
        </table>
        <script>
            const ctx = document.getElementById('scoreChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'Anomaly Score', data: [], borderColor: '#38bdf8', tension: 0.2, fill: false },
                    { label: 'Threshold', data: [], borderColor: '#ef4444', borderDash: [5, 5], pointRadius: 0, fill: false }
                ]},
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: '#334155' } }, x: { grid: { color: '#334155' } } } }
            });

            async function refreshData() {
                const res = await fetch('/history?limit=30');
                const data = await res.json();
                if (!data.length) return;

                const latest = data[0];
                document.getElementById('sys-status').innerHTML = latest.is_anomaly ? '<span style="color:#ef4444">CRITICAL</span>' : '<span style="color:#10b981">NORMAL</span>';
                document.getElementById('latest-score').innerText = latest.anomaly_score;
                document.getElementById('top-sensor').innerText = latest.top_contributing_sensor;
                document.getElementById('alert-count').innerText = data.filter(d => d.is_anomaly).length;

                const chronological = [...data].reverse();
                chart.data.labels = chronological.map(d => '#' + d.id);
                chart.data.datasets[0].data = chronological.map(d => d.anomaly_score);
                chart.data.datasets[1].data = chronological.map(d => d.threshold);
                chart.update();

                const tbody = document.getElementById('log-table');
                tbody.innerHTML = data.slice(0, 10).map(r => `
                    <tr>
                        <td>#${r.id}</td>
                        <td>${r.timestamp.slice(0, 19).replace('T', ' ')}</td>
                        <td><span class="${r.is_anomaly ? 'badge-crit' : 'badge-norm'}">${r.status}</span></td>
                        <td>${r.anomaly_score}</td>
                        <td>${r.threshold}</td>
                        <td>${r.top_contributing_sensor}</td>
                        <td>${r.sensor_deviation_zscore}</td>
                    </tr>
                `).join('');

                const driftRes = await fetch('/drift?window_size=30');
                const drift = await driftRes.json();
                document.getElementById('drift-status').innerText = `Data Drift Status: ${drift.status}`;
            }
            setInterval(refreshData, 1500);
            refreshData();
        </script>
    </body>
    </html>
    """