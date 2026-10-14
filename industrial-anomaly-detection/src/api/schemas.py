from pydantic import BaseModel, Field


class SensorReadingRequest(BaseModel):
    """
    اسکیمای اعتبارسنجی داده‌های ورودی سنسورها مطابق با دیتاست SKAB.
    """
    Accelerometer1RMS: float = Field(..., description="Root Mean Square of Accelerometer 1")
    Accelerometer2RMS: float = Field(..., description="Root Mean Square of Accelerometer 2")
    Current: float = Field(..., description="Electrical current in Amperes")
    Pressure: float = Field(..., description="Loop pressure")
    Temperature: float = Field(..., description="Fluid temperature")
    Thermocouple: float = Field(..., description="Thermocouple reading")
    Voltage: float = Field(..., description="Electrical voltage")
    Volume_Flow_RateRMS: float = Field(..., alias="Volume Flow RateRMS", description="Fluid volume flow rate RMS")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Accelerometer1RMS": 0.026,
                "Accelerometer2RMS": 0.040,
                "Current": 1.30,
                "Pressure": 0.054,
                "Temperature": 79.2,
                "Thermocouple": 26.8,
                "Voltage": 230.5,
                "Volume Flow RateRMS": 32.0,
            }
        },
    }


class AnomalyPredictionResponse(BaseModel):
    """
    اسکیمای خروجی سرویس تشخیص ناهنجاری.
    """
    record_id: int | None = None
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    status: str
    top_contributing_sensor: str
    sensor_deviation_zscore: float