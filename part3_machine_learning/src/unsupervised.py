"""Unsupervised learning: k-means clustering and association rules.

Association rules are mined with a small Apriori-style search written here (itemsets of up
to three items, support and confidence and lift computed explicitly). mlxtend is not required.
"""
import itertools
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.features import chronological_split  # noqa: E402
from src.tracking import Tracker  # noqa: E402

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["hour_sin", "hour_cos", "severity_score", "traffic_volume"]
K_RANGE = range(2, 9)
MIN_SUPPORT = 0.01
CONSEQUENT = "congestion=High"


# ---------- k-means ----------
def choose_k(scaled: np.ndarray) -> tuple[pd.DataFrame, int]:
    rows = []
    rng = np.random.default_rng(config.SEED)
    sample_idx = rng.choice(len(scaled), size=min(5000, len(scaled)), replace=False)
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(scaled)
        sil = silhouette_score(scaled[sample_idx], km.labels_[sample_idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.debug("k=%d inertia=%.0f silhouette=%.3f", k, km.inertia_, sil)
    table = pd.DataFrame(rows)
    best_k = int(table.loc[table["silhouette"].idxmax(), "k"])
    logger.info("Silhouette-maximising k = %d", best_k)
    return table, best_k


def daypart(hour: float) -> str:
    h = int(round(hour)) % 24
    if h <= 5:
        return "overnight"
    if h <= 9:
        return "morning peak"
    if h <= 14:
        return "midday"
    if h <= 18:
        return "evening peak"
    return "late evening"


def name_cluster(part: pd.DataFrame) -> str:
    """Plain-language name from the cluster's own profile (time, day type, traffic, weather)."""
    mean_angle = np.arctan2(part["hour_sin"].mean(), part["hour_cos"].mean())
    mean_hour = (mean_angle % (2 * np.pi)) * 24 / (2 * np.pi)
    when = daypart(mean_hour)
    weekday_share = (part["is_weekend"] == 0).mean()
    day = "weekday" if weekday_share >= 0.6 else "weekend" if weekday_share <= 0.4 else "mixed-day"
    vol = part["traffic_volume"].mean()
    level = "quiet" if vol < 1500 else "moderate" if vol < 3500 else "busy" if vol < 5000 else "peak"
    adverse = part["adverse_weather"].mean()
    weather = "adverse weather" if adverse >= 0.3 else "clear-to-cloudy" if part["severity_score"].mean() <= 1.5 else "mixed weather"
    return f"{when.capitalize()}, {day}, {level}, {weather}"


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df.assign(cluster=labels)
    rows = []
    for c, part in df.groupby("cluster"):
        rows.append({
            "cluster": c,
            "name": name_cluster(part),
            "hours": len(part),
            "share_of_hours": len(part) / len(df),
            "mean_volume": part["traffic_volume"].mean(),
            "congestion_rate": part["congested"].mean(),
            "weekday_share": (part["is_weekend"] == 0).mean(),
            "adverse_weather_share": part["adverse_weather"].mean(),
            "mean_temp_K": part["temp"].mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_volume")


def run_kmeans(df: pd.DataFrame, tracker: Tracker) -> pd.DataFrame:
    # Scaler is fitted on the training years only (2017 and earlier, outage window excluded),
    # then applied to every row, so 2018 never shapes the scaling.
    train, _ = chronological_split(df)
    scaler = StandardScaler().fit(train[CLUSTER_FEATURES].to_numpy())
    scaled = scaler.transform(df[CLUSTER_FEATURES].to_numpy())
    table, best_k = choose_k(scaled)
    table.to_csv(config.METRIC_DIR / "kmeans_k_selection.csv", index=False)
    _plot_k_selection(table, best_k)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=config.SEED).fit(scaled)
    profiles = cluster_profiles(df, km.labels_)
    profiles.to_csv(config.METRIC_DIR / "cluster_profiles.csv", index=False)
    for _, row in profiles.iterrows():
        logger.info("Cluster %d: %s (%.1f%% of hours, mean %.0f vehicles/h)",
                    row["cluster"], row["name"], 100 * row["share_of_hours"], row["mean_volume"])
    _plot_cluster_profiles(profiles)
    tracker.log_run(
        run_name=f"kmeans-k{best_k}",
        params={"k": best_k, "features": ",".join(CLUSTER_FEATURES), "seed": config.SEED, "scaled": True},
        metrics={"silhouette": float(table.loc[table["k"] == best_k, "silhouette"].iloc[0]),
                 "inertia": float(km.inertia_)},
    )
    return profiles


# ---------- association rules ----------
def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    items = pd.DataFrame(index=df.index)
    dp = df["hour"].map(lambda h: daypart(h))
    for label in ["overnight", "morning peak", "midday", "evening peak", "late evening"]:
        items[f"time={label}"] = (dp == label)
    items["day=weekend"] = df["is_weekend"] == 1
    items["day=weekday"] = df["is_weekend"] == 0
    items["holiday=yes"] = df["is_holiday"] == 1
    w = df["weather_main"]
    items["weather=clear"] = w == "Clear"
    items["weather=clouds"] = w == "Clouds"
    items["weather=rain or drizzle"] = w.isin(["Rain", "Drizzle"])
    items["weather=snow or thunder"] = w.isin(["Snow", "Thunderstorm", "Squall"])
    items["weather=low visibility"] = df["low_visibility"] == 1
    items[CONSEQUENT] = df["congestion_band"] == "High"
    return items


def mine_rules(items: pd.DataFrame, max_antecedent: int = 2, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    n = len(items)
    antecedent_items = [c for c in items.columns if c != CONSEQUENT]
    consequent = items[CONSEQUENT].to_numpy()
    p_cons = consequent.mean()
    rows = []
    for size in range(1, max_antecedent + 1):
        for combo in itertools.combinations(antecedent_items, size):
            mask = items[list(combo)].all(axis=1).to_numpy()
            supp_a = mask.mean()
            if supp_a < min_support:
                continue
            supp_ac = (mask & consequent).mean()
            if supp_ac < min_support:
                continue
            conf = supp_ac / supp_a
            rows.append({
                "antecedent": " AND ".join(combo),
                "consequent": CONSEQUENT,
                "support": supp_ac,
                "confidence": conf,
                "lift": conf / p_cons,
                "rows": int(mask.sum()),
            })
    rules = pd.DataFrame(rows).sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    logger.info("Mined %d rules predicting %s from %d transactions (base rate %.3f)",
                len(rules), CONSEQUENT, n, p_cons)
    return rules


def describe_rule(row: pd.Series, base_rate: float) -> str:
    return (f"When {row['antecedent']}, high congestion (>{config.CONGESTION_THRESHOLD} veh/h) occurs in "
            f"{row['confidence']:.0%} of hours, {row['lift']:.2f} times the overall rate of {base_rate:.0%}.")


def run_rules(df: pd.DataFrame, tracker: Tracker) -> pd.DataFrame:
    items = build_transactions(df)
    rules = mine_rules(items)
    base = items[CONSEQUENT].mean()
    top = rules.head(10).copy()
    top["plain_language"] = [describe_rule(r, base) for _, r in top.iterrows()]
    rules.to_csv(config.METRIC_DIR / "association_rules_all.csv", index=False)
    top.to_csv(config.METRIC_DIR / "association_rules_top10.csv", index=False)
    for _, r in top.head(3).iterrows():
        logger.info("Rule: %s", r["plain_language"])
    tracker.log_run(
        run_name="association-rules",
        params={"min_support": MIN_SUPPORT, "max_antecedent": 2, "consequent": CONSEQUENT},
        metrics={"n_rules": len(rules), "top_lift": float(rules["lift"].max())},
    )
    return top


def _plot_k_selection(table: pd.DataFrame, best_k: int) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(table["k"], table["inertia"], marker="o", color="#2b6cb0", label="Inertia (elbow)")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia")
    ax2 = ax1.twinx()
    ax2.plot(table["k"], table["silhouette"], marker="s", color="#dd6b20", label="Silhouette")
    ax2.axvline(best_k, color="grey", ls="--")
    ax2.set_ylabel("Silhouette")
    ax1.set_title(f"Choosing k: silhouette peaks at k = {best_k}")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "kmeans_elbow_silhouette.png", dpi=130)
    plt.close(fig)


def _plot_cluster_profiles(profiles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"C{r.cluster}: {r['name']}" for _, r in profiles.iterrows()]
    ax.barh(labels, profiles["mean_volume"], color="#2b6cb0")
    ax.set_xlabel("Mean vehicles per hour")
    ax.set_title("k-means clusters, named from their profiles")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "kmeans_cluster_profiles.png", dpi=130)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    tracker = Tracker("traffic-unsupervised")
    profiles = run_kmeans(df, tracker)
    rules = run_rules(df, tracker)
    return {"profiles": profiles, "rules": rules}
