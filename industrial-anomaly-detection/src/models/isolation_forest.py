from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from src.data.loader import load_skab_file, split_features_and_labels


class SensorIsolationForest:
    """
    آشکارساز ناهنجاری چندمتغیره با الگوریتم Isolation Forest.
    فضای رفتار نرمال سنسورها را صرفاً از داده‌های استاندارد و بدون خرابی یاد می‌گیرد.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "SensorIsolationForest":
        self.feature_names = list(X.columns)
        self.model.fit(X)
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        محاسبه امتیاز ناهنجاری (Anomaly Score).
        کتابخانه scikit-learn به طور پیش‌فرض مقادیر منفی‌تر را به عنوان ناهنجارتر خروجی می‌دهد.
        علامت را معکوس می‌کنیم تا عدد بزرگ‌تر نشان‌دهنده ریسک خرابی بالاتر باشد.
        """
        raw_scores = self.model.score_samples(X)
        return -raw_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        تبدیل تصمیم مدل به فرمت باینری استاندارد:
        صفر برای وضعیت عادی (Normal) و یک برای ناهنجاری (Anomaly).
        """
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def save(self, filepath: Path) -> None:
        """
        ذخیره وزن‌ها و تنظیمات مدل نهایی به صورت یک آرتیفکت joblib.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "SensorIsolationForest":
        return joblib.load(filepath)


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    artifacts_dir = Path("models")

    # 1. Load data
    train_df = load_skab_file(raw_dir / "anomaly-free.csv")
    test_df = load_skab_file(raw_dir / "valve1_0.csv")

    X_train, _ = split_features_and_labels(train_df)
    X_test, y_test = split_features_and_labels(test_df)

    # 2. Train Isolation Forest strictly on normal baseline
    iso_forest = SensorIsolationForest(contamination=0.02, n_estimators=150, random_state=42)
    iso_forest.fit(X_train)

    # 3. Evaluate on test set
    y_pred = iso_forest.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=== Isolation Forest Model Evaluation ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], digits=4))

    # 4. Save trained artifact for production API inference
    artifact_path = artifacts_dir / "isolation_forest.joblib"
    iso_forest.save(artifact_path)
    print(f"\nModel artifact saved to: {artifact_path}")