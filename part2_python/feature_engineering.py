"""Stage 2: turn the cleaned data into machine-learning-ready features."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import LOW_VISIBILITY_WEATHER, PRECIPITATION_WEATHER, SEVERE_WEATHER

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "processed" / "traffic_clean.csv"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "traffic_features.csv"


def _rel(path):
    """Show paths relative to this folder in log messages (shorter and portable)."""
    try:
        return Path(path).resolve().relative_to(BASE_DIR)
    except ValueError:
        return path

RUSH_HOURS = [7, 8, 16, 17, 18]  # from Part 1: peaks at 07:00 and 16:00-17:00

CONGESTION_LABELS = ["Low", "Medium", "High", "Severe"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["date_time"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # Monday=0 ... Sunday=6
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (df["hour"].isin(RUSH_HOURS) & (df["is_weekend"] == 0)).astype(int)
    # Cyclical encoding: 23:00 and 00:00 are neighbours, so use sine/cosine
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    logger.info("Added time features: hour, day_of_week, month, year, is_weekend, is_rush_hour, "
                "and sine/cosine pairs for hour, day of week and month")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = (df["holiday"] != "No Holiday").astype(int)
    df["temp_c"] = df["temp"] - 273.15
    df["is_raining"] = (df["rain_1h"] > 0).astype(int)
    df["is_snowing"] = (df["snow_1h"] > 0).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)
    df["is_precipitation"] = df["weather_main"].isin(PRECIPITATION_WEATHER).astype(int)
    df["is_severe_weather"] = df["weather_main"].isin(SEVERE_WEATHER).astype(int)
    # One-hot encode the weather category
    dummies = pd.get_dummies(df["weather_main"], prefix="weather", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    logger.info("Added weather features: is_holiday, temp_c, is_raining, is_snowing, is_low_visibility, "
                "is_precipitation, is_severe_weather and %d one-hot weather columns", dummies.shape[1])
    return df


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale continuous variables so they are on comparable ranges."""
    df = df.copy()
    # Standardisation (z-score) for temperature
    temp_mean, temp_std = df["temp_c"].mean(), df["temp_c"].std()
    logger.debug("temp_c mean=%.4f std=%.4f (used for z-score)", temp_mean, temp_std)
    df["temp_c_scaled"] = (df["temp_c"] - temp_mean) / temp_std
    # Min-max scaling to 0-1 for cloud cover and hourly rain (log first: very skewed)
    clouds_min, clouds_max = df["clouds_all"].min(), df["clouds_all"].max()
    logger.debug("clouds_all min=%s max=%s (used for min-max)", clouds_min, clouds_max)
    df["clouds_scaled"] = (df["clouds_all"] - clouds_min) / (clouds_max - clouds_min)
    log_rain = np.log1p(df["rain_1h"])
    logger.debug("log1p(rain_1h) max=%.4f (used for min-max)", log_rain.max())
    df["rain_scaled"] = log_rain / log_rain.max() if log_rain.max() > 0 else 0.0
    logger.info("Added scaled features: temp_c_scaled (z-score), clouds_scaled and rain_scaled (min-max 0-1)")
    return df


def add_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    """Congestion category from traffic_volume quartiles.

    Logic: compute the 25th, 50th and 75th percentiles of traffic_volume.
      volume <= Q1        -> Low
      Q1 < volume <= Q2   -> Medium
      Q2 < volume <= Q3   -> High
      volume > Q3         -> Severe
    So each category holds roughly a quarter of all hours, and thresholds are
    driven by the data rather than picked by hand.
    """
    df = df.copy()
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    logger.debug("Congestion quartile thresholds: Q1=%.1f Q2=%.1f Q3=%.1f", q1, q2, q3)

    def bucket(volume: float) -> str:
        if volume <= q1:
            return "Low"
        elif volume <= q2:
            return "Medium"
        elif volume <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = df["traffic_volume"].apply(bucket)
    df["congestion_category"] = pd.Categorical(df["congestion_category"], categories=CONGESTION_LABELS, ordered=True)
    counts = df["congestion_category"].value_counts().reindex(CONGESTION_LABELS).to_dict()
    logger.info("Added congestion_category (quartile-based) with counts %s", counts)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    rows, cols = df.shape
    logger.info("Feature engineering started: shape before = %d rows x %d columns", rows, cols)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_features(df)
    df = add_congestion_category(df)
    rows, cols = df.shape
    logger.info("Feature engineering finished: shape after = %d rows x %d columns", rows, cols)
    return df


def run_feature_engineering(in_path: Path = CLEAN_PATH, out_path: Path = FEATURES_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(in_path, parse_dates=["date_time"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cleaned file not found: {in_path}. Run pipeline.py first.") from exc
    df = engineer_features(df)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    except OSError:
        logger.error("Could not save features to %s", out_path, exc_info=True)
        raise
    logger.info("Saved feature dataset to %s", _rel(out_path))
    return df
