"""Pure prediction logic shared by the CLI and the API. No web framework is needed to test it."""
import logging

import joblib
import pandas as pd

from src import config
from src.features import build_frame, regression_feature_names

logger = logging.getLogger(__name__)

REQUIRED = ["temp", "clouds_all", "weather_main"]
DEFAULTS = {"rain_1h": 0.0, "snow_1h": 0.0, "holiday": "No Holiday"}


def load_regressor(path=config.MODEL_DIR / "random_forest_regressor.joblib"):
    return joblib.load(path)


def load_classifier(path=config.MODEL_DIR / "random_forest_classifier.joblib"):
    return joblib.load(path)


def payload_to_frame(payload: dict) -> pd.DataFrame:
    """Validate a request and build the one-row frame the feature code expects."""
    missing = [k for k in ["datetime", *REQUIRED] if payload.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")
    if payload["weather_main"] not in config.SEVERITY_RANK:
        raise ValueError(f"Unknown weather_main '{payload['weather_main']}'")
    try:
        when = pd.to_datetime(payload["datetime"], format="%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as exc:
        raise ValueError("datetime must look like 'YYYY-MM-DD HH:MM:SS'") from exc
    row = {"date_time": when, **DEFAULTS, **{k: payload[k] for k in payload if k != "datetime"}}
    temp = float(row["temp"])
    if temp <= 0:
        raise ValueError("temp must be in kelvin and above 0")
    return pd.DataFrame([row])


def predict_volume(model, payload: dict) -> dict:
    frame = build_frame(payload_to_frame(payload))
    X = frame[regression_feature_names()]
    predicted = float(model.predict(X)[0])
    band = "Low" if predicted < config.BAND_LOW_MAX else "Medium" if predicted <= config.CONGESTION_THRESHOLD else "High"
    return {"predicted_volume": round(predicted), "congestion_band": band, "model": "random_forest_regressor v1.0.0"}


def predict_high_risk(model, payload: dict) -> float:
    from src.features import CLASSIFIER_FEATURES
    frame = build_frame(payload_to_frame(payload))
    return float(model.predict_proba(frame[CLASSIFIER_FEATURES])[0, 1])
