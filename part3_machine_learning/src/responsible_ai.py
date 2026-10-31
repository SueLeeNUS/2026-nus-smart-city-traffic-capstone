"""Responsible AI: data coverage, group performance, proxy-label disclosure.

There is no protected attribute in this dataset, so fairness is assessed across the situations
where the model's errors could fall unevenly: day type, time of day and weather. Results are
descriptive. They do not certify fairness.
"""
import logging

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import CLASSIFIER_FEATURES, chronological_split, regression_feature_names

logger = logging.getLogger(__name__)


def coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(df["date_time"].dt.year)
    table = pd.DataFrame({
        "hours_recorded": grp.size(),
        "months_with_data": grp["date_time"].apply(lambda s: s.dt.month.nunique()),
        "mean_volume": grp["traffic_volume"].mean().round(0),
    })
    table["coverage_of_full_year"] = (table["hours_recorded"] / 8760).round(3)
    table.to_csv(config.METRIC_DIR / "coverage_by_year.csv")
    logger.warning("Coverage: years with under 60%% of hours recorded: %s",
                   table.index[table["coverage_of_full_year"] < 0.6].tolist())
    return table


def _clf_group_row(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "group": name, "rows": len(y), "positive_rate": y.mean(),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def group_performance(df: pd.DataFrame) -> pd.DataFrame:
    _, test = chronological_split(df)
    clf = joblib.load(config.MODEL_DIR / "random_forest_classifier.joblib")
    reg = joblib.load(config.MODEL_DIR / "random_forest_regressor.joblib")
    proba = clf.predict_proba(test[CLASSIFIER_FEATURES])[:, 1]
    vol_pred = reg.predict(test[regression_feature_names()])
    test = test.assign(proba=proba, vol_pred=vol_pred, abs_err=np.abs(vol_pred - test["traffic_volume"]),
                       signed_err=vol_pred - test["traffic_volume"])

    groups = {
        "weekday": test["is_weekend"] == 0,
        "weekend": test["is_weekend"] == 1,
        "daytime 06-18": test["hour"].between(6, 18),
        "night 19-05": ~test["hour"].between(6, 18),
        "clear or cloudy": ~test["weather_main"].isin(["Thunderstorm", "Squall", "Snow", "Fog", "Mist", "Haze", "Smoke", "Rain", "Drizzle"]),
        "rain, drizzle or storm": test["weather_main"].isin(["Rain", "Drizzle", "Thunderstorm", "Squall"]),
    }
    rows = []
    for name, mask in groups.items():
        sub = test[mask]
        row = _clf_group_row(name, sub["high_risk"].to_numpy(), sub["proba"].to_numpy())
        row["volume_MAE"] = sub["abs_err"].mean()
        row["volume_mean_error (bias)"] = sub["signed_err"].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(config.METRIC_DIR / "bias_report_2018.csv", index=False)
    logger.info("Group performance written for %d groups", len(table))
    return table


def run(df: pd.DataFrame) -> dict:
    coverage = coverage_by_year(df)
    groups = group_performance(df)
    return {"coverage": coverage, "groups": groups}
