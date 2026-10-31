"""Part 2 pipeline stage: raw CSV -> one-row-per-hour cleaned file (the Part 3 input).

Part 3 consumes only data/processed/traffic_clean.csv. This module reproduces that stage so
the Part 3 folder can be run from the raw file on its own.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw file. Typed exceptions only, and schema is validated before anything else."""
    try:
        # "None" in the holiday column is a real value meaning "no holiday" (as in Part 1 SQL),
        # so pandas' default NA parsing must be switched off or it silently becomes NaN.
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except FileNotFoundError:
        logger.error("Raw data file not found: %s", path)
        raise
    except pd.errors.ParserError as exc:
        logger.error("Could not parse raw CSV %s: %s", path, exc)
        raise
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Schema check failed, missing columns: {missing}")
    empty = {c: int(df[c].isna().sum()) for c in EXPECTED_COLUMNS if df[c].isna().any()}
    if empty:
        logger.warning("Empty cells found (set to missing): %s", empty)
    logger.info("Loaded %d raw rows x %d columns from %s", df.shape[0], df.shape[1], path.name)
    return df


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Trim categoricals, parse timestamps and enforce numeric types."""
    df = df.copy()
    for col in ["holiday", "weather_main", "weather_description"]:
        df[col] = df[col].fillna("None" if col == "holiday" else "unknown").astype(str).str.strip()
    df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    logger.debug("Standardised dtypes: %s", df.dtypes.astype(str).to_dict())
    return df


def collapse_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hour. Keep the most severe weather label (severity order in config)."""
    before = len(df)
    exact_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df = df.assign(_rank=df["weather_main"].map(config.SEVERITY_RANK).fillna(len(config.SEVERITY_ORDER)))
    df = df.sort_values(["date_time", "_rank", "weather_description"], kind="mergesort")
    hourly = df.drop_duplicates(subset="date_time", keep="first").drop(columns="_rank")
    if not hourly["date_time"].is_unique:
        raise AssertionError("date_time is still not unique after hourly collapse")
    logger.info(
        "Hourly collapse: %d raw rows -> %d hours (%d exact duplicates removed first)",
        before, len(hourly), exact_dupes,
    )
    return hourly.reset_index(drop=True)


def fill_holidays_across_day(df: pd.DataFrame) -> pd.DataFrame:
    """The raw holiday flag sits only on 00:00. Fill it across the whole calendar day."""
    named = df.loc[df["holiday"] != "None", ["date_time", "holiday"]].copy()
    named["day"] = named["date_time"].dt.normalize()
    if (named.groupby("day")["holiday"].nunique() > 1).any():
        logger.warning("Some calendar days carry more than one holiday name; keeping the first")
    day_map = named.groupby("day")["holiday"].first()
    flagged_before = int((df["holiday"] != "None").sum())
    day = df["date_time"].dt.normalize()
    df = df.assign(holiday=day.map(day_map).fillna("None"))
    flagged_after = int((df["holiday"] != "None").sum())
    logger.warning("Holiday fill: %d flagged hours -> %d holiday hours", flagged_before, flagged_after)
    return df


def handle_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Set impossible readings to missing, then impute each with an explicit monthly-median loop."""
    df = df.copy()
    bad = {
        "temp": df["temp"] <= 0.0,                              # 0 K is physically impossible
        "rain_1h": df["rain_1h"] > config.MAX_PLAUSIBLE_RAIN_MM,
        "traffic_volume": df["traffic_volume"] == 0,            # sensor dropout
    }
    for col, mask in bad.items():
        df[f"imputed_{col}"] = mask
        logger.warning("Impossible %s readings: %d set to missing before imputation", col, int(mask.sum()))
        df.loc[mask, col] = np.nan

    year_month = df["date_time"].dt.to_period("M")
    for col in bad:
        missing_rows = df.index[df[col].isna()]
        for idx in missing_rows:
            same_month = df.loc[(year_month == year_month.loc[idx]) & df[col].notna(), col]
            fill = same_month.median() if len(same_month) else df[col].median()
            df.at[idx, col] = fill
        logger.info("Imputed %d %s values with their monthly median", len(missing_rows), col)
    return df


def log_quartiles(df: pd.DataFrame) -> None:
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).round(0).tolist()
    logger.debug("Traffic quartile thresholds: Q1=%s Q2=%s Q3=%s", q1, q2, q3)
    ref = config.QUARTILE_FALLBACK
    shifts = [abs(a - b) for a, b in zip((q1, q2, q3), ref)]
    if max(shifts) > 1:
        logger.warning("Quartiles %s differ from reference %s by more than 1 vehicle", (q1, q2, q3), ref)
    elif any(shifts):
        logger.info(
            "Quartiles %s vs reference %s: Q1 moves by 1 vehicle because the two zero-traffic "
            "readings were imputed (the reference was computed before imputation)", (q1, q2, q3), ref,
        )


def run(raw_path: Path = config.RAW_PATH, out_path: Path = config.REBUILD_CHECK_PATH) -> pd.DataFrame:
    """Legacy rebuild of the Part 2 stage, kept only as a check. It writes a separate file and
    is NOT used by run_all.py, so it cannot replace Part 2's cleaned file."""
    raw = load_raw(raw_path)
    df = standardise(raw)
    df = collapse_to_hourly(df)
    df = fill_holidays_across_day(df)
    df = handle_impossible_values(df)
    log_quartiles(df)
    df = df.sort_values("date_time").reset_index(drop=True)
    if len(df) != df["date_time"].nunique():
        raise AssertionError("Cleaned file is not one row per hour")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    logger.info("Wrote %d hourly rows to %s", len(df), out_path)
    return df


def load_clean(path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    """Read the cleaned hourly file (the only input Part 3 modelling code uses)."""
    df = pd.read_csv(path, parse_dates=["date_time"], keep_default_na=False, na_values=[""])
    logger.debug("Loaded cleaned file with %d rows", len(df))
    return df
