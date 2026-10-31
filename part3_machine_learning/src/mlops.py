"""MLOps: model version table, drift simulation over 2018 and a PASS / WARN / ALERT reporter.

Drift is simulated by replaying 2018 month by month against the training period (<=2017).
Feature drift uses the Population Stability Index (PSI). Performance drift compares monthly MAE
with the out-of-sample MAE measured on 2017 by a model trained on 2012-2016 only.
"""
import logging
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from src import config  # noqa: E402
from src.features import chronological_split, regression_feature_names  # noqa: E402
from src.tracking import Tracker  # noqa: E402

logger = logging.getLogger(__name__)

DRIFT_FEATURES = ["temp", "rain_1h", "clouds_all", "traffic_volume"]
MAE_ALERT_RISE = 0.25  # alert if monthly MAE is 25% or more above the 2017 reference


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, smooth: float = 0.5) -> float:
    """Population Stability Index against a reference distribution.

    Low-cardinality features (e.g. cloud cover percent) use one bin per distinct value.
    Continuous features use reference quantile bins. Proportions are smoothed so an empty
    reference bin cannot produce an infinite term.
    """
    uniq = np.unique(reference)
    if len(uniq) <= bins:
        mids = (uniq[1:] + uniq[:-1]) / 2
        edges = np.r_[-np.inf, mids, np.inf]
    else:
        inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.r_[-np.inf, inner, np.inf]
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    k = len(ref_counts)
    ref_p = (ref_counts + smooth) / (len(reference) + smooth * k)
    cur_p = (cur_counts + smooth) / (len(current) + smooth * k)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_status(value: float) -> str:
    if value >= config.PSI_ALERT:
        return "ALERT"
    if value >= config.PSI_WARN:
        return "WARN"
    return "PASS"


def reference_mae_2017(df: pd.DataFrame) -> float:
    """Out-of-sample MAE on 2017 for a model trained on 2012-2016 only."""
    years = df["date_time"].dt.year
    tr, va = df[years <= 2016], df[years == 2017]
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=config.SEED
    )
    cols = regression_feature_names()
    model.fit(tr[cols], tr["traffic_volume"])
    mae = float(np.mean(np.abs(model.predict(va[cols]) - va["traffic_volume"])))
    logger.info("Reference MAE on 2017 (trained <=2016): %.1f vehicles/h", mae)
    return mae


def monitor_2018(df: pd.DataFrame, tracker: Tracker) -> pd.DataFrame:
    train, test = chronological_split(df)
    model = joblib.load(config.MODEL_DIR / "random_forest_regressor.joblib")
    cols = regression_feature_names()
    ref_mae = reference_mae_2017(df)
    test = test.assign(pred=model.predict(test[cols]))

    rows = []
    train_month = train["date_time"].dt.month
    for month, batch in test.groupby(test["date_time"].dt.month):
        # Compare like with like: each 2018 month is measured against the same calendar month in
        # the training years, so a cold January is not reported as drift.
        same_month = train[train_month == month]
        row = {"month": f"2018-{month:02d}", "rows": len(batch), "reference_rows": len(same_month)}
        for feat in DRIFT_FEATURES:
            value = psi(same_month[feat].to_numpy(), batch[feat].to_numpy())
            row[f"psi_{feat}"] = round(value, 4)
        mae = float(np.mean(np.abs(batch["pred"] - batch["traffic_volume"])))
        row["mae"] = round(mae, 1)
        row["mae_vs_2017_reference"] = round((mae - ref_mae) / ref_mae, 3)
        # Cloud cover is checked separately: its 2018 shift matches a change in how the code was
        # recorded (the share of code "1" rises in every month), not a change in the weather.
        # A month with no rain recorded at all is a data gap, not evidence about rain.
        row["cloud_coding_check"] = "CHECK" if row["psi_clouds_all"] >= config.PSI_ALERT else "ok"
        rain_gap = (batch["rain_1h"] > 0).sum() == 0 and (same_month["rain_1h"] > 0).mean() > 0.01
        row["rain_data_gap"] = "GAP" if rain_gap else "ok"
        trusted = [f for f in DRIFT_FEATURES if f != "clouds_all" and not (f == "rain_1h" and rain_gap)]
        worst_psi = max(row[f"psi_{f}"] for f in trusted)
        if worst_psi >= config.PSI_ALERT or row["mae_vs_2017_reference"] >= MAE_ALERT_RISE:
            status = "ALERT"
        elif worst_psi >= config.PSI_WARN or row["mae_vs_2017_reference"] >= MAE_ALERT_RISE / 2:
            status = "WARN"
        else:
            status = "PASS"
        row["status"] = status
        if status == "ALERT":
            logger.warning("Drift ALERT for %s: worst PSI %.3f, MAE change %+.1f%%",
                           row["month"], worst_psi, 100 * row["mae_vs_2017_reference"])
        else:
            logger.info("Drift %s for %s: worst PSI %.3f, MAE change %+.1f%%",
                        status, row["month"], worst_psi, 100 * row["mae_vs_2017_reference"])
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(config.METRIC_DIR / "drift_monthly_2018.csv", index=False)
    _plot_drift(report, ref_mae)
    tracker.log_run(
        run_name="drift-simulation-2018",
        params={"reference": "<=2017", "monitor": "2018 monthly", "psi_warn": config.PSI_WARN,
                "psi_alert": config.PSI_ALERT},
        metrics={"reference_mae_2017": ref_mae, "alert_months": int((report["status"] == "ALERT").sum())},
    )
    return report


def model_version_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per registered model: what it is, what it was trained on, how it scored."""
    today = date.today().isoformat()
    reg = pd.read_csv(config.METRIC_DIR / "regression_metrics.csv", index_col=0)
    clf = pd.read_csv(config.METRIC_DIR / "classification_metrics.csv", index_col=0)
    entries = [
        ("v1.0.0", "random_forest_regressor", "regression: traffic_volume",
         f"MAE {reg.loc['random_forest', 'MAE']:.0f}, R2 {reg.loc['random_forest', 'R2']:.3f}", "production (serves API)"),
        ("v1.0.0", "ridge_regressor", "regression: traffic_volume",
         f"MAE {reg.loc['ridge', 'MAE']:.0f}, R2 {reg.loc['ridge', 'R2']:.3f}", "baseline, retained"),
        ("v1.0.0", "random_forest_classifier", "classification: high_risk",
         f"PR-AUC {clf.loc['random_forest', 'pr_auc']:.3f}, recall {clf.loc['random_forest', 'recall']:.3f}", "production"),
        ("v1.0.0", "logistic_regression_classifier", "classification: high_risk",
         f"PR-AUC {clf.loc['logistic_regression', 'pr_auc']:.3f}, recall {clf.loc['logistic_regression', 'recall']:.3f}", "baseline, retained"),
    ]
    table = pd.DataFrame(entries, columns=["version", "model", "task", "test_2018", "status"])
    table.insert(0, "registered", today)
    table["train_data"] = f"hourly rows dated <={config.TRAIN_END_YEAR}"
    table.to_csv(config.METRIC_DIR / "model_versions.csv", index=False)
    logger.info("Wrote model version table with %d entries", len(table))
    return table


def _plot_drift(report: pd.DataFrame, ref_mae: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(report))
    ax.plot(x, report["mae"], marker="o", color="#2b6cb0", label="Monthly MAE")
    ax.axhline(ref_mae, color="grey", ls="--", label="2017 out-of-sample MAE (reference)")
    alerts = report["status"].eq("ALERT").to_numpy()
    if alerts.any():
        ax.scatter(x[alerts], report["mae"].to_numpy()[alerts], color="#c53030", zorder=3, label="ALERT")
    ax.set_xticks(x, report["month"], rotation=30)
    ax.set_title("Drift simulation, 2018: monthly error against the training period")
    ax.set_ylabel("MAE (vehicles/h)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "drift_monthly_mae.png", dpi=130)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    tracker = Tracker("traffic-mlops")
    table = model_version_table(df)
    report = monitor_2018(df, tracker)
    return {"versions": table, "drift": report}
