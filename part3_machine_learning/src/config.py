"""Single source of truth for Part 3: random seed, paths, label definitions and thresholds.

Every module imports its constants from here so Part 1 (SQL), Part 2 (pipeline) and
Part 3 (models) agree on the same definitions.
"""
from pathlib import Path

# Reproducibility: one seed, imported and set everywhere (splits, k-means, model fits).
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
# Part 2's cleaned file is the single input to Part 3 (copied here unchanged from Part 2 output).
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv"
REBUILD_CHECK_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_clean_rebuild_check.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
TRACKING_DIR = PROJECT_ROOT / "mlruns_local"

# Severity ordering used to collapse duplicate hours (most severe weather wins).
SEVERITY_ORDER = [
    "Thunderstorm", "Squall", "Snow", "Rain", "Drizzle", "Fog",
    "Mist", "Haze", "Smoke", "Clouds", "Clear",
]
SEVERITY_RANK = {label: rank for rank, label in enumerate(SEVERITY_ORDER)}

# Proxy-label sets. The brief references both without defining them; these are our definitions.
SEVERE_WEATHER = frozenset({"Thunderstorm", "Squall", "Snow"})
LOW_VISIBILITY = frozenset({"Fog", "Mist", "Haze", "Smoke"})

# Congestion definitions (three are required and kept distinct).
CONGESTION_THRESHOLD = 5500          # fixed, policy-facing: volume above 5,500
BAND_LOW_MAX = 4500                  # dashboard bands: Low <= 4,500 < Medium <= 5,500 < High
QUARTILE_FALLBACK = (1248, 3427, 4952)  # reference quartiles, logged and checked at run time

# High-risk traffic threshold: the brief's fixed value (the all-years median). This is the one
# exception to the 2017-and-earlier rule; the training-years median is 3,415 (12 vehicles lower).
HIGH_RISK_TRAFFIC_THRESHOLD = 3427

# Cleaning rules (Part 2 stage, reproduced here so Part 3 can run from the raw file).
MAX_PLAUSIBLE_RAIN_MM = 100.0        # hourly rainfall above this is treated as a sensor error

# Holiday label used by Part 2 for "not a holiday". Older text "None" is also accepted on input.
NO_HOLIDAY = "No Holiday"

# Suspect sensor/road-outage window from Part 2 (22 Jul 22:00 to 24 Jul 17:00, 2016, 44 hours).
# Excluded from training and test so it cannot shape the model or the scores.
ANOMALY_START = "2016-07-22 22:00:00"
ANOMALY_END = "2016-07-24 17:00:00"

# Chronological evaluation split: train on 2012-2017, test on 2018 (never a random split).
TRAIN_END_YEAR = 2017
TEST_YEAR = 2018

# Population Stability Index thresholds for the drift reporter.
PSI_WARN = 0.10
PSI_ALERT = 0.20

for _folder in (FIGURE_DIR, METRIC_DIR, MODEL_DIR, LOG_DIR, TRACKING_DIR, CLEAN_PATH.parent):
    _folder.mkdir(parents=True, exist_ok=True)
