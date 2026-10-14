import time
import json
import urllib.request
from pathlib import Path
from src.data.loader import load_skab_file, split_features_and_labels


def stream_sensor_data(file_path: Path, api_url: str = "http://127.0.0.1:8000/predict", max_rows: int = 40, delay: float = 0.5):
    """
    شبیه‌سازی ارسال زنده داده‌های سنسور از فایل آزمایش به سرویس FastAPI.
    """
    df = load_skab_file(file_path)
    X, y = split_features_and_labels(df)

    # انتخاب ترکیبی از رکوردهای سالم و ناهنجار (بازه اطراف شروع خرابی در دیتاست)
    subset_X = X.iloc[580 : 580 + max_rows]
    subset_y = y.iloc[580 : 580 + max_rows]

    print(f"=== Starting Sensor Stream Simulation ({len(subset_X)} records) ===")
    for idx, (_, row) in enumerate(subset_X.iterrows()):
        payload = row.to_dict()
        actual_label = int(subset_y.iloc[idx])

        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            print(
                f"[{idx+1:02d}] Actual={actual_label} | Pred={res['status']:8s} | "
                f"Score={res['anomaly_score']:.4f} | TopSensor={res['top_contributing_sensor']}"
            )
        time.sleep(delay)


if __name__ == "__main__":
    stream_sensor_data(Path("data/raw/valve1_0.csv"))