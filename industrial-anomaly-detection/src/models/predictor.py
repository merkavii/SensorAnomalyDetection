from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_skab_file, split_features_and_labels
from src.models.isolation_forest import SensorIsolationForest


class AnomalyPredictor:
    """
    موتور استنتاج (Inference Engine) برای بارگذاری مدل و پیش‌بینی لحظه‌ای.
    علاوه بر تشخیص ناهنجاری، سنسوری که بیشترین انحراف آماری را دارد نیز مشخص می‌کند.
    """

    def __init__(self, model: SensorIsolationForest, sensor_means: pd.Series, sensor_stds: pd.Series):
        """
        مقداردهی اولیه موتور پیش‌بینی با مدل آموزش‌دیده و پارامترهای آماری پایه.
        """
        self.model = model
        self.sensor_means = sensor_means # میانگین هر ۸ سنسور در وضعیت کاملاً سالم (محاسبه‌شده فقط از روی دیتای آموزش).
        self.sensor_stds = sensor_stds.replace(0, 1e-6)
        self.threshold = float(-self.model.model.offset_)

    def predict_single(self, sensor_reading: Dict[str, float]) -> Dict[str, Any]:
        """
        دریافت یک رکورد از مقادیر سنسورها و بازگرداندن نتیجه کامل تشخیص و ریشه‌یابی.
        """
        df = pd.DataFrame([sensor_reading])[self.model.feature_names] # دیکشنری ورودی را به یک سطر DataFrame تبدیل می‌کند و ترتیب ستون‌ها را دقیقاً مطابق با ترتیب زمان آموزش (feature_names) می‌چیند

        is_anomaly = bool(self.model.predict(df)[0] == 1)
        anomaly_score = float(self.model.predict_score(df)[0])

        # محاسبه میزان انحراف هر سنسور برای پیدا کردن مقصر اصلی (Root Cause Approximation)
        z_scores = np.abs((df.iloc[0] - self.sensor_means) / self.sensor_stds) # برای رکورد جدید محاسبه می‌کند که هر سنسور چند برابرِ انحراف معیار (Z-Score) از میانگینِ سالمِ خودش فاصله گرفته
        top_sensor = str(z_scores.idxmax()) # سپس با idxmax() اسم سنسوری که بیشترین پرش را داشته به عنوان top_contributing_sensor استخراج می‌کند.
        top_sensor_z = float(z_scores.max())

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(self.threshold, 4),
            "status": "CRITICAL" if is_anomaly else "NORMAL",
            "top_contributing_sensor": top_sensor,
            "sensor_deviation_zscore": round(top_sensor_z, 2),
        }

    def save(self, filepath: Path) -> None:
        """
        ذخیره کل پایپ‌لاین استنتاج روی دیسک.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "AnomalyPredictor":
        """
        بارگذاری پایپ‌لاین استنتاج آماده به کار برای استفاده در FastAPI.
        """
        return joblib.load(filepath)


if __name__ == "__main__":
    # ایمپورت با مسیر کامل پکیج تا joblib آدرس دقیق ماژول را ثبت کند و به __main__ وابسته نشود
    from src.models.isolation_forest import SensorIsolationForest as PkgIsoForest
    from src.models.predictor import AnomalyPredictor as PkgAnomalyPredictor

    raw_dir = Path("data/raw")
    artifacts_dir = Path("models")

    train_df = load_skab_file(raw_dir / "anomaly-free.csv")
    X_train, _ = split_features_and_labels(train_df)

    # آموزش و ذخیره با مسیر کامل ماژول
    model_path = artifacts_dir / "isolation_forest.joblib"
    iso_model = PkgIsoForest(contamination=0.02, n_estimators=150, random_state=42)
    iso_model.fit(X_train)
    iso_model.save(model_path)

    # ساخت و ذخیره Predictor نهایی با مسیر کامل ماژول
    predictor = PkgAnomalyPredictor(
        model=iso_model,
        sensor_means=X_train.mean(),
        sensor_stds=X_train.std(),
    )
    predictor_path = artifacts_dir / "predictor.joblib"
    predictor.save(predictor_path)

    # تست سریع روی یک نمونه نرمال
    sample_input = X_train.iloc[0].to_dict()
    result = predictor.predict_single(sample_input)
    print("=== Predictor Pipeline Saved & Verified ===")
    print(f"Artifact saved to: {predictor_path}")
    print(f"Sample Inference Output: {result}")