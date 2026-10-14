from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from src.data.loader import load_skab_file, split_features_and_labels


class ZScoreAnomalyDetector:
    """
    Statistical baseline detector using Maximum Absolute Z-score across sensors.
    Learns sensor means and standard deviations strictly from normal data.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "ZScoreAnomalyDetector":
        self.means = X.mean()
        # Avoid division by zero with small epsilon
        self.stds = X.std().replace(0, 1e-6)
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("Model is not fitted yet.")

        # Compute absolute z-score per sensor: |x - mean| / std
        z_scores = np.abs((X - self.means) / self.stds)
        # Anomaly score is the worst-case deviation across any sensor
        max_z_scores = z_scores.max(axis=1).to_numpy()
        return max_z_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)


if __name__ == "__main__":
    raw_dir = Path("data/raw")

    # 1. Load data
    train_df = load_skab_file(raw_dir / "anomaly-free.csv")
    test_df = load_skab_file(raw_dir / "valve1_0.csv")

    X_train, _ = split_features_and_labels(train_df)
    X_test, y_test = split_features_and_labels(test_df)

    # 2. Fit Baseline strictly on normal training data
    baseline = ZScoreAnomalyDetector(threshold=3.0)
    baseline.fit(X_train)

    # 3. Evaluate on test set
    y_pred = baseline.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=== Baseline Model (Z-Score, Threshold=3.0) Evaluation ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], digits=4))