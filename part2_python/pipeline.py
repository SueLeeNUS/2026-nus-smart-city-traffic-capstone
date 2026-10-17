"""Stage 1: load, validate and clean the raw Metro Interstate traffic CSV.

Run on its own:   python pipeline.py
(main.py runs this stage plus feature engineering and figures.)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

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
    changed = int((before.ne(df[["holiday", "weather_main", "weather_description"]])).any(axis=1).sum())
    # 'None' -> 'No Holiday' is an intentional relabel, so count it separately
    relabelled = int((before["holiday"] == "None").sum())
    logger.info(
        "Standardised categoricals: %d holiday 'None' values relabelled 'No Holiday' "
        "(a real value, not missing); %d rows otherwise changed in capitalisation/whitespace",
        relabelled, max(changed - relabelled, 0),
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


def resolve_repeated_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Several rows can describe the same hour (extra weather labels).
    Traffic volume is identical within an hour, so we keep one row per hour:
    the first one in file order."""
    n = int(df["date_time"].duplicated().sum())
    if n:
        # Check our assumption that traffic is the same within a repeated hour
        varying = int((df.groupby("date_time")["traffic_volume"].nunique() > 1).sum())
        if varying:
            logger.warning("%d repeated hours have differing traffic_volume; keeping first row", varying)
        logger.warning(
            "Dropped %d rows: repeated timestamp (extra weather label for the same hour; "
            "kept the first row per hour)", n)
        df = df.drop_duplicates(subset="date_time", keep="first")
    else:
        logger.info("No repeated timestamps found")
    return df


def _monthly_median_impute(df: pd.DataFrame, col: str, bad_mask: pd.Series, reason: str) -> pd.DataFrame:
    """Replace flagged values with the median of VALID values in the same
    calendar month (group-by-group, loop over months). Falls back to the
    overall median if a month has no valid values."""
    df = df.copy()
    n = int(bad_mask.sum())
    if n == 0:
        logger.info("No %s found in %s", reason, col)
        return df
    months = df["date_time"].dt.month
    overall_median = float(df.loc[~bad_mask, col].median())
    imputed_counts = {}
    for month in sorted(months[bad_mask].unique()):
        valid_in_month = df.loc[(months == month) & ~bad_mask, col]
        if len(valid_in_month) > 0:
            replacement = float(valid_in_month.median())
        else:
            replacement = overall_median
        target = bad_mask & (months == month)
        df.loc[target, col] = replacement
        imputed_counts[int(month)] = (int(target.sum()), round(replacement, 3))
        logger.debug("%s: month %d -> %d rows set to monthly median %.3f",
                     col, month, int(target.sum()), replacement)
    logger.warning("Imputed %d rows in '%s' with the same-month median: %s", n, col, reason)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Impossible values -> monthly median (see _monthly_median_impute)."""
    df = _monthly_median_impute(
        df, "temp", (df["temp"] < MIN_TEMP_K) | (df["temp"] > MAX_TEMP_K),
        f"impossible temperature (outside {MIN_TEMP_K}-{MAX_TEMP_K} K, e.g. 0 K sensor error)")
    df = _monthly_median_impute(
        df, "rain_1h", (df["rain_1h"] < 0) | (df["rain_1h"] > MAX_RAIN_MM_PER_HOUR),
        f"implausible rainfall (above {MAX_RAIN_MM_PER_HOUR} mm/hour or negative)")
    df = _monthly_median_impute(
        df, "snow_1h", (df["snow_1h"] < 0) | (df["snow_1h"] > MAX_SNOW_MM_PER_HOUR),
        f"implausible snowfall (above {MAX_SNOW_MM_PER_HOUR} mm/hour or negative)")
    df = _monthly_median_impute(
        df, "clouds_all", (df["clouds_all"] < 0) | (df["clouds_all"] > 100),
        "cloud cover outside 0-100 %")
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
    df = resolve_repeated_timestamps(df)
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
