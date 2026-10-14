from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.loader import load_skab_file, split_features_and_labels
from src.models.predictor import AnomalyPredictor


def test_data_loader_shapes():
    """
    بررسی صحت بارگذاری فایل‌های داده و وجود هر ۸ سنسور استاندارد.
    """
    df = load_skab_file(Path("data/raw/anomaly-free.csv"))
    X, y = split_features_and_labels(df)
    assert X.shape[1] == 8
    assert len(X) == len(y)
    assert "Volume Flow RateRMS" in X.columns


def test_predictor_inference_output():
    """
    بررسی صحت خروجی موتور استنتاج و وجود فیلدهای کلیدی تصمیم‌گیری و ریشه‌یابی.
    """
    predictor = AnomalyPredictor.load(Path("models/predictor.joblib"))
    sample_payload = {
        "Accelerometer1RMS": 0.026,
        "Accelerometer2RMS": 0.040,
        "Current": 1.30,
        "Pressure": 0.054,
        "Temperature": 79.2,
        "Thermocouple": 26.8,
        "Voltage": 230.5,
        "Volume Flow RateRMS": 32.0,
    }
    res = predictor.predict_single(sample_payload)
    assert "is_anomaly" in res
    assert "threshold" in res
    assert res["status"] in ["NORMAL", "CRITICAL"]
    assert res["top_contributing_sensor"] in sample_payload


def test_api_health_and_predict_endpoints():
    """
    تست یکپارچه اندپوینت‌های FastAPI و ثبت رکورد در دیتابیس SQLite.
    """
    with TestClient(app) as client:
        # 1. Health endpoint
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["model_loaded"] is True

        # 2. Predict endpoint (Normal reading)
        payload = {
            "Accelerometer1RMS": 0.026,
            "Accelerometer2RMS": 0.040,
            "Current": 1.30,
            "Pressure": 0.054,
            "Temperature": 79.2,
            "Thermocouple": 26.8,
            "Voltage": 230.5,
            "Volume Flow RateRMS": 32.0,
        }
        pred_res = client.post("/predict", json=payload)
        assert pred_res.status_code == 200
        data = pred_res.json()
        assert data["record_id"] is not None
        assert "threshold" in data

        # 3. History endpoint
        hist_res = client.get("/history?limit=5")
        assert hist_res.status_code == 200
        assert len(hist_res.json()) >= 1