"""Supervised learning: two regressors (traffic_volume) and two classifiers (high_risk).

Evaluation is chronological (train <= 2017, test 2018). Classification leads with precision,
recall and PR-AUC; accuracy is reported only beside the never-high-risk baseline.
"""
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score,
    r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.features import CLASSIFIER_FEATURES, chronological_split, regression_feature_names  # noqa: E402
from src.tracking import Tracker  # noqa: E402

logger = logging.getLogger(__name__)


def naive_hour_of_week(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: the training-period mean for the same weekday and hour."""
    profile = train.groupby(["dayofweek", "hour"])["traffic_volume"].mean()
    keys = pd.MultiIndex.from_arrays([test["dayofweek"], test["hour"]])
    return profile.reindex(keys).fillna(train["traffic_volume"].mean()).to_numpy()


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_regression(df: pd.DataFrame, tracker: Tracker) -> dict:
    train, test = chronological_split(df)
    cols = regression_feature_names()
    X_tr, y_tr = train[cols], train["traffic_volume"]
    X_te, y_te = test[cols], test["traffic_volume"]

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=0.5,
            n_jobs=-1, random_state=config.SEED,
        ),
    }
    results = {}
    baseline_pred = naive_hour_of_week(train, test)
    results["naive_hour_of_week"] = regression_metrics(y_te, baseline_pred)
    logger.info("Regression baseline (hour-of-week mean): %s", results["naive_hour_of_week"])

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = regression_metrics(y_te, pred)
        path = config.MODEL_DIR / f"{name}_regressor.joblib"
        joblib.dump(model, path)
        logger.info("Regressor %s test metrics: %s", name, {k: round(v, 3) for k, v in results[name].items()})
        tracker.log_run(
            run_name=f"regressor-{name}",
            params={"model": name, "train_years": f"<={config.TRAIN_END_YEAR}", "test_year": config.TEST_YEAR,
                    "seed": config.SEED, "n_features": len(cols)},
            metrics=results[name],
            artifacts=[path],
        )

    rf = models["random_forest"]
    sample = test.sample(n=min(2000, len(test)), random_state=config.SEED)
    imp = permutation_importance(
        rf, sample[cols], sample["traffic_volume"], n_repeats=5,
        scoring="neg_mean_absolute_error", random_state=config.SEED, n_jobs=-1,
    )
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    importance.to_csv(config.METRIC_DIR / "regression_permutation_importance.csv", header=["mae_increase"])
    _plot_importance(importance.head(12), "Regressor: permutation importance (MAE increase, 2018)",
                     config.FIGURE_DIR / "regression_permutation_importance.png")

    _plot_pred_vs_actual(test, rf.predict(X_te), config.FIGURE_DIR / "rf_week_actual_vs_predicted.png")
    pd.DataFrame(results).T.to_csv(config.METRIC_DIR / "regression_metrics.csv")
    return {"results": results, "importance": importance}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run_classification(df: pd.DataFrame, tracker: Tracker) -> dict:
    train, test = chronological_split(df)
    cols = CLASSIFIER_FEATURES
    X_tr, y_tr = train[cols], train["high_risk"]
    X_te, y_te = test[cols], test["high_risk"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.SEED),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.SEED,
        ),
    }
    never_risk_accuracy = 1 - y_te.mean()
    logger.info(
        "High-risk test positives: %d of %d (%.1f%%); never-high-risk accuracy would be %.1f%%",
        int(y_te.sum()), len(y_te), 100 * y_te.mean(), 100 * never_risk_accuracy,
    )
    results = {}
    probas = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        probas[name] = proba
        results[name] = classification_metrics(y_te, proba)
        path = config.MODEL_DIR / f"{name}_classifier.joblib"
        joblib.dump(model, path)
        logger.info("Classifier %s: precision=%.3f recall=%.3f PR-AUC=%.3f",
                    name, results[name]["precision"], results[name]["recall"], results[name]["pr_auc"])
        tracker.log_run(
            run_name=f"classifier-{name}",
            params={"model": name, "label": "high_risk", "threshold": 0.5,
                    "class_weight": "balanced", "seed": config.SEED},
            metrics={k: v for k, v in results[name].items()},
            artifacts=[path],
        )

    results["never_high_risk_baseline"] = {"accuracy": never_risk_accuracy, "pr_auc": y_te.mean()}

    # Ablation: how much of the signal is time of day alone? Weather is excluded entirely.
    from src.features import TIME_FEATURES
    ablation = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
        n_jobs=-1, random_state=config.SEED,
    )
    ablation.fit(train[TIME_FEATURES], y_tr)
    abl_proba = ablation.predict_proba(test[TIME_FEATURES])[:, 1]
    results["random_forest_time_only"] = classification_metrics(y_te, abl_proba)
    logger.info("Ablation (time features only): PR-AUC=%.3f precision=%.3f",
                results["random_forest_time_only"]["pr_auc"], results["random_forest_time_only"]["precision"])
    pd.DataFrame(results).T.to_csv(config.METRIC_DIR / "classification_metrics.csv")
    _plot_pr_curves(y_te, probas, config.FIGURE_DIR / "classifier_precision_recall.png")

    rf = models["random_forest"]
    sample = test.sample(n=min(2000, len(test)), random_state=config.SEED)
    imp = permutation_importance(
        rf, sample[cols], sample["high_risk"], n_repeats=5,
        scoring="average_precision", random_state=config.SEED, n_jobs=-1,
    )
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    importance.to_csv(config.METRIC_DIR / "classifier_permutation_importance.csv", header=["pr_auc_drop"])
    _plot_importance(importance, "Classifier: permutation importance (PR-AUC drop, 2018)",
                     config.FIGURE_DIR / "classifier_permutation_importance.png")
    return {"results": results, "importance": importance, "probas": probas, "y_test": y_te}


def _plot_importance(series: pd.Series, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    series.sort_values().plot.barh(ax=ax, color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel("Drop in score when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved figure %s", path)


def _plot_pred_vs_actual(test: pd.DataFrame, pred, path) -> None:
    week = test.assign(pred=pred)
    week = week[(week["date_time"] >= "2018-03-05") & (week["date_time"] < "2018-03-12")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week["date_time"], week["traffic_volume"], label="Actual", color="#1a202c", lw=1)
    ax.plot(week["date_time"], week["pred"], label="Random forest", color="#dd6b20", lw=1)
    ax.set_title("One week in March 2018: actual vs predicted hourly traffic")
    ax.set_ylabel("Vehicles per hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved figure %s", path)


def _plot_pr_curves(y_true, probas: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ax.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y_true, proba):.2f})")
    ax.axhline(y_true.mean(), color="grey", ls="--", label="Never-high-risk baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("High-risk classifiers, 2018 test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved figure %s", path)


def run(df: pd.DataFrame) -> dict:
    tracker = Tracker("traffic-supervised")
    reg = run_regression(df, tracker)
    clf = run_classification(df, tracker)
    return {"regression": reg, "classification": clf}
