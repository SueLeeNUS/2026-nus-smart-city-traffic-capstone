"""Stage 1: load, validate and clean the raw Metro Interstate traffic CSV.

Run on its own:   python pipeline.py
(main.py runs this stage plus feature engineering and figures.)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import WEATHER_SEVERITY_ORDER

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "traffic_clean.csv"


def _rel(path):
    """Show paths relative to this folder in log messages (shorter and portable)."""
    try:
        return Path(path).resolve().relative_to(BASE_DIR)
    except ValueError:
        return path

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

# Physically plausible limits used to flag impossible sensor readings.
MIN_TEMP_K = 200.0        # about -73 C; 0 K is a sensor error
MAX_TEMP_K = 330.0        # about 57 C
MAX_RAIN_MM_PER_HOUR = 500.0   # far above any real hourly rainfall (record ~305 mm)
MAX_SNOW_MM_PER_HOUR = 500.0
MAX_TRAFFIC = 10_000      # well above the observed maximum (7,280)


class PipelineError(Exception):
    """Raised when the pipeline cannot continue."""


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV. The text 'None' in `holiday` is a real value
    (meaning 'not a holiday'), so we stop pandas turning it into NaN."""
    try:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except FileNotFoundError as exc:
        raise PipelineError(f"Raw file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise PipelineError(f"Raw file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise PipelineError(f"Raw file could not be parsed: {path}") from exc
    except PermissionError as exc:
        raise PipelineError(f"No permission to read: {path}") from exc
    logger.info("Loaded raw file %s: %d rows, %d columns", _rel(path), *df.shape)
    return df


# --------------------------------------------------------------------------
# 2. Validate schema
# --------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """All expected columns must exist before anything else happens."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise PipelineError(f"Schema check failed, missing columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        logger.warning("Schema check: %d unexpected extra column(s) kept: %s", len(extra), extra)
    logger.info("Schema check passed: all %d expected columns present", len(EXPECTED_COLUMNS))


# --------------------------------------------------------------------------
# 3. Clean (one function per step, one log message per step)
# --------------------------------------------------------------------------
def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and fix capitalisation of text columns."""
    df = df.copy()
    before = df[["holiday", "weather_main", "weather_description"]].copy()
    df["holiday"] = df["holiday"].astype(str).str.strip().replace({"None": "No Holiday", "": "No Holiday"})
    df["weather_main"] = df["weather_main"].astype(str).str.strip().str.title()
    df["weather_description"] = df["weather_description"].astype(str).str.strip().str.lower()
    # 'None' -> 'No Holiday' is an intentional relabel, so it is counted separately from text clean-up
    relabelled = int((before["holiday"] == "None").sum())
    text_cols = ["weather_main", "weather_description"]
    text_changed = int(before[text_cols].ne(df[text_cols]).any(axis=1).sum())
    logger.info(
        "Standardised categoricals: %d holiday 'None' values relabelled 'No Holiday' "
        "(a real value, not missing); %d rows had weather text changed in capitalisation/whitespace",
        relabelled, text_changed,
    )
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time; drop rows that cannot be parsed."""
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    bad = int(df["date_time"].isna().sum())
    if bad:
        logger.warning("Dropped %d rows: date_time could not be parsed", bad)
        df = df.dropna(subset=["date_time"])
    else:
        logger.info("Parsed date_time: all %d values valid (range %s to %s)",
                    len(df), df["date_time"].min(), df["date_time"].max())
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure numeric columns are numbers; unreadable values become NaN and are dropped."""
    df = df.copy()
    numeric_cols = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[numeric_cols].isna().any(axis=1)
    n_bad = int(bad_rows.sum())
    if n_bad:
        logger.warning("Dropped %d rows: non-numeric or missing values in numeric columns", n_bad)
        df = df.loc[~bad_rows]
    else:
        logger.info("Numeric check: all numeric columns valid, no missing values")
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n = int(df.duplicated().sum())
    if n:
        logger.warning("Dropped %d rows: exact duplicate of another row", n)
        df = df.drop_duplicates()
    else:
        logger.info("No exact duplicate rows found")
    return df


def collect_holiday_dates(df: pd.DataFrame) -> dict:
    """Remember which calendar dates are holidays (before rows are collapsed).
    In the raw file only the 00:00 row of a holiday carries its name."""
    flagged = df[df["holiday"] != "No Holiday"]
    holiday_by_date = {}
    for ts, name in zip(flagged["date_time"], flagged["holiday"]):
        holiday_by_date.setdefault(ts.date(), name)
    logger.debug("Holiday dates found in raw data: %d", len(holiday_by_date))
    return holiday_by_date


def collapse_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """GRAIN RULE: keep exactly one row per hour.

    The raw file lists an hour once per weather description, with identical traffic.
    We keep the row with the most severe weather_main (order in config.py), breaking
    ties alphabetically by weather_description, the same rule as the SQL view
    traffic_hourly in Part 1."""
    n_extra = int(df["date_time"].duplicated().sum())
    if n_extra:
        varying = int((df.groupby("date_time")["traffic_volume"].nunique() > 1).sum())
        if varying:
            logger.warning("%d repeated hours have differing traffic_volume; keeping most severe row", varying)
        hours_with_copies = int((df.groupby("date_time").size() > 1).sum())
        logger.debug("%d hours appear more than once in the data", hours_with_copies)
        rank_map = {w: i for i, w in enumerate(WEATHER_SEVERITY_ORDER)}
        rank = df["weather_main"].map(rank_map).fillna(len(WEATHER_SEVERITY_ORDER))
        ordered = (df.assign(_severity=rank)
                     .sort_values(["date_time", "_severity", "weather_description"], kind="mergesort"))
        df = ordered.drop_duplicates(subset="date_time", keep="first").drop(columns="_severity")
        logger.warning(
            "Dropped %d rows: repeated hour (kept one row per hour, the most severe weather_main; "
            "traffic volume is identical across the copies)", n_extra)
    else:
        logger.info("No repeated timestamps found")
    return df


def verify_unique_hours(df: pd.DataFrame) -> None:
    """Fail loudly if the one-row-per-hour rule did not hold."""
    if df["date_time"].duplicated().any():
        raise PipelineError("date_time is not unique after collapsing to hourly rows")
    logger.info("Verified: date_time is unique, %d hourly rows", len(df))


def fill_holiday_across_day(df: pd.DataFrame, holiday_by_date: dict) -> pd.DataFrame:
    """Copy each holiday's name to every hour of that calendar date.
    In the raw file only 00:00 is flagged, so the flag is almost useless as a feature."""
    df = df.copy()
    before = int((df["holiday"] != "No Holiday").sum())
    dates = df["date_time"].dt.date
    filled = dates.map(holiday_by_date)
    df["holiday"] = filled.fillna("No Holiday")
    after = int((df["holiday"] != "No Holiday").sum())
    n_dates = int(df.loc[df["holiday"] != "No Holiday", "date_time"].dt.date.nunique())
    if after != before:
        logger.warning(
            "Modified %d rows: holiday label filled across the whole calendar day "
            "(flagged hours %d -> %d, across %d holiday dates)", after - before, before, after, n_dates)
    else:
        logger.info("Holiday label already covers whole days (%d hours, %d dates)", after, n_dates)
    return df


def _median_impute(df: pd.DataFrame, col: str, bad_mask: pd.Series, reason: str,
                   by_hour: bool = False) -> pd.DataFrame:
    """Replace flagged values with the median of VALID values in the same calendar
    month (and, if by_hour, the same hour of day). Done group by group in a loop.
    Falls back to the overall median if a group has no valid values."""
    df = df.copy()
    n = int(bad_mask.sum())
    if n == 0:
        logger.info("No %s found in %s", reason, col)
        return df
    months = df["date_time"].dt.month
    hours = df["date_time"].dt.hour
    key = months * 100 + hours if by_hour else months
    overall_median = float(df.loc[~bad_mask, col].median())
    for group in sorted(key[bad_mask].unique()):
        valid_in_group = df.loc[(key == group) & ~bad_mask, col]
        replacement = float(valid_in_group.median()) if len(valid_in_group) > 0 else overall_median
        if pd.api.types.is_integer_dtype(df[col]):
            replacement = int(round(replacement))  # whole-number columns stay whole numbers
        target = bad_mask & (key == group)
        df.loc[target, col] = replacement
        label = f"month {group // 100} hour {group % 100:02d}" if by_hour else f"month {group}"
        logger.debug("%s: %s -> %d rows set to median %.3f", col, label, int(target.sum()), replacement)
    scope = "same-month-and-hour" if by_hour else "same-month"
    logger.warning("Imputed %d rows in '%s' with the %s median: %s", n, col, scope, reason)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Impossible values -> monthly median (see _median_impute)."""
    df = _median_impute(
        df, "temp", (df["temp"] < MIN_TEMP_K) | (df["temp"] > MAX_TEMP_K),
        f"impossible temperature (outside {MIN_TEMP_K}-{MAX_TEMP_K} K, e.g. 0 K sensor error)")
    df = _median_impute(
        df, "rain_1h", (df["rain_1h"] < 0) | (df["rain_1h"] > MAX_RAIN_MM_PER_HOUR),
        f"implausible rainfall (above {MAX_RAIN_MM_PER_HOUR} mm/hour or negative)")
    df = _median_impute(
        df, "snow_1h", (df["snow_1h"] < 0) | (df["snow_1h"] > MAX_SNOW_MM_PER_HOUR),
        f"implausible snowfall (above {MAX_SNOW_MM_PER_HOUR} mm/hour or negative)")
    df = _median_impute(
        df, "clouds_all", (df["clouds_all"] < 0) | (df["clouds_all"] > 100),
        "cloud cover outside 0-100 %")
    df = _median_impute(
        df, "traffic_volume", df["traffic_volume"] == 0,
        "zero vehicles for a full hour on a major interstate (sensor dropout)", by_hour=True)
    bad_traffic = (df["traffic_volume"] < 0) | (df["traffic_volume"] > MAX_TRAFFIC)
    if int(bad_traffic.sum()):
        logger.warning("Dropped %d rows: traffic_volume outside 0-%d", int(bad_traffic.sum()), MAX_TRAFFIC)
        df = df.loc[~bad_traffic]
    else:
        logger.info("traffic_volume check: all values within 0-%d", MAX_TRAFFIC)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order."""
    df = standardise_categoricals(df)
    df = parse_datetimes(df)
    df = coerce_numeric(df)
    df = remove_exact_duplicates(df)
    holiday_by_date = collect_holiday_dates(df)
    df = collapse_to_hourly(df)
    verify_unique_hours(df)
    df = fill_holiday_across_day(df, holiday_by_date)
    df = handle_outliers(df)
    df = df.sort_values("date_time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Public entry for this stage
# --------------------------------------------------------------------------
def run_pipeline(raw_path: Path = RAW_PATH, out_path: Path = CLEAN_PATH) -> pd.DataFrame:
    logger.info("Cleaning stage started")
    df = load_raw(raw_path)
    validate_schema(df)
    df = clean(df)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    except OSError as exc:
        raise PipelineError(f"Could not write cleaned file {out_path}: {exc}") from exc
    logger.info("Saved cleaned data to %s: %d rows, %d columns", _rel(out_path), *df.shape)
    return df


def main() -> int:
    from logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Clean the raw traffic CSV.")
    parser.add_argument("--debug", action="store_true", help="show DEBUG messages")
    parser.add_argument("--log-file", default=str(BASE_DIR / "pipeline.log"))
    args = parser.parse_args()
    setup_logging(args.log_file, debug=args.debug)
    try:
        run_pipeline()
    except PipelineError:
        logger.error("Pipeline cannot complete; exiting", exc_info=True)
        return 1
    except Exception:  # last-resort guard so the program exits gracefully
        logger.error("Unexpected pipeline failure; exiting", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    # Re-import under the name "pipeline" so log lines show a clear module name
    from pipeline import main as _main

    sys.exit(_main())
