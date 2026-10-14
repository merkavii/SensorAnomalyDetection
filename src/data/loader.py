from pathlib import Path
from typing import Tuple
import pandas as pd


# ستون‌های برچسب که نباید به عنوان ویژگی به مدل داده شوند
LABEL_COLUMNS = ["anomaly", "changepoint"]


def load_skab_file(file_path: Path) -> pd.DataFrame:
    """یک فایل CSV از دیتاست SKAB را می‌خواند و ستون زمان را به ایندکس تبدیل می‌کند."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # فایل‌های SKAB با سمی‌کالن (;) از هم جدا شده‌اند
    df = pd.read_csv(file_path, sep=";")

    # تبدیل ستون datetime به ایندکس زمانی سری زمانی
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

    return df


def split_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """ویژگی‌های سنسورها را از برچسب ناهنجاری جدا می‌کند."""
    feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    X = df[feature_cols].copy()

    # اگر ستون anomaly وجود داشته باشد برمی‌گرداند، در غیر این صورت صفر می‌گذارد
    if "anomaly" in df.columns:
        y = df["anomaly"].astype(int)
    else:
        y = pd.Series(0, index=df.index, name="anomaly")

    return X, y


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    
    # تست لود دیتای نرمال
    normal_df = load_skab_file(raw_dir / "anomaly-free.csv")
    X_norm, _ = split_features_and_labels(normal_df)
    
    # تست لود دیتای تستی با خرابی
    test_df = load_skab_file(raw_dir / "valve1_0.csv")
    X_test, y_test = split_features_and_labels(test_df)

    print("=== Data Loaded Successfully ===")
    print(f"Normal training samples: {len(X_norm)}")
    print(f"Test evaluation samples: {len(X_test)}")
    print(f"Sensor features count: {len(X_norm.columns)}")
    print("Feature list:")
    for col in X_norm.columns:
        print(f"  - {col}")
    anomaly_rate = (y_test.mean() * 100)
    print(f"Test set anomaly rate: {anomaly_rate:.2f}%")