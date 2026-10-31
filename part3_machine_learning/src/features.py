"""Feature engineering for Part 3. Every model takes its feature list from here.

Leakage rules (documented in the report):
- The proxy label high_risk is built from weather_main and traffic_volume, so neither is a
  classifier input. snow_1h is excluded because it nearly identifies the Snow category.
- The regressor predicts traffic_volume, so the target never enters its inputs.
"""
import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

WEATHER_CATEGORIES = config.SEVERITY_ORDER

TIME_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "is_holiday", "month_sin", "month_cos",
]
REGRESSION_BASE = TIME_FEATURES + ["temp", "rain_1h", "snow_1h", "clouds_all"]
CLASSIFIER_FEATURES = TIME_FEATURES + ["temp", "rain_1h", "clouds_all"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["date_time"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = (~df["holiday"].isin([config.NO_HOLIDAY, "None"])).astype(int)
    # Cyclical encodings so 23:00 sits next to 00:00 and December next to January.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severe_weather"] = df["weather_main"].isin(config.SEVERE_WEATHER).astype(int)
    df["low_visibility"] = df["weather_main"].isin(config.LOW_VISIBILITY).astype(int)
    df["adverse_weather"] = ((df["severe_weather"] + df["low_visibility"]) > 0).astype(int)
    rank = df["weather_main"].map(config.SEVERITY_RANK)
    # 0 = clear ... 10 = thunderstorm. Unknown labels score 0.
    df["severity_score"] = (len(config.SEVERITY_ORDER) - 1 - rank).fillna(0).astype(int)
    return df


def training_quartiles(df: pd.DataFrame) -> tuple:
    """Quartile cut-offs of traffic_volume computed on training years only (2017 and earlier,
    outage window excluded). No 2018 value can move a threshold."""
    years = df["date_time"].dt.year
    mask = (years <= config.TRAIN_END_YEAR) & (in_anomaly_window(df) == 0)
    q1, q2, q3 = df.loc[mask, "traffic_volume"].quantile([0.25, 0.5, 0.75]).tolist()
    return q1, q2, q3


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Targets and congestion categories. Only applied when traffic_volume is present.
    Quartile cut-offs come from training years only. The high-risk traffic threshold is the
    fixed value 3,427 from the brief (see README)."""
    df = df.copy()
    v = df["traffic_volume"]
    df["congested"] = (v > config.CONGESTION_THRESHOLD).astype(int)
    # Dashboard bands, written as explicit conditions so the boundaries match Part 1 exactly:
    # Low below 4,500; Medium 4,500 to 5,500 inclusive; High above 5,500.
    df["congestion_band"] = np.select(
        [v < config.BAND_LOW_MAX, v <= config.CONGESTION_THRESHOLD],
        ["Low", "Medium"],
        default="High",
    )
    q1, q2, q3 = training_quartiles(df)
    df["quartile_band"] = pd.cut(
        v, bins=[-np.inf, q1, q2, q3, np.inf], labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    df["high_risk"] = (
        (df["adverse_weather"] == 1) & (v > config.HIGH_RISK_TRAFFIC_THRESHOLD)
    ).astype(int)
    return df


def in_anomaly_window(df: pd.DataFrame) -> pd.Series:
    """True for hours inside the Part 2 outage window (kept in the data, never modelled)."""
    t = df["date_time"]
    return ((t >= pd.Timestamp(config.ANOMALY_START)) & (t <= pd.Timestamp(config.ANOMALY_END))).astype(int)


def weather_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed category list, so one-row inference produces the same columns as training."""
    cat = df["weather_main"].astype(pd.CategoricalDtype(WEATHER_CATEGORIES))
    dummies = pd.get_dummies(cat, prefix="w", dtype=int)
    dummies.index = df.index
    return dummies


def regression_feature_names() -> list:
    return REGRESSION_BASE + [f"w_{c}" for c in WEATHER_CATEGORIES]


def build_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Cleaned hourly rows -> full feature frame used by every Part 3 model."""
    df = add_time_features(clean)
    df = add_weather_features(df)
    if "traffic_volume" in df.columns:
        df = add_labels(df)
    df["in_anomaly_window"] = in_anomaly_window(df)
    df = pd.concat([df, weather_dummies(df)], axis=1)
    logger.debug("Feature frame built: %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def chronological_split(df: pd.DataFrame):
    """Train on years up to TRAIN_END_YEAR, test on TEST_YEAR. Never a random split."""
    years = df["date_time"].dt.year
    window = in_anomaly_window(df).astype(bool)
    train = df[(years <= config.TRAIN_END_YEAR) & ~window].copy()
    test = df[(years == config.TEST_YEAR) & ~window].copy()
    logger.info(
        "Chronological split: train %d rows (<=%d), test %d rows (%d); %d rows in the 22-24 July 2016 "
        "outage window excluded from both",
        len(train), config.TRAIN_END_YEAR, len(test), config.TEST_YEAR, int(window.sum()),
    )
    return train, test
