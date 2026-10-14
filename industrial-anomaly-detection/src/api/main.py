from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query

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
    description="Real-time multivariate sensor anomaly detection and root-cause isolation API.",
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
def get_prediction_history(limit: int = Query(default=20, ge=1, le=100)):
    """
    دریافت تاریخچه آخرین پیش‌بینی‌ها و هشدارهای ثبت‌شده در پایگاه داده.
    """
    return fetch_recent_predictions(limit=limit)