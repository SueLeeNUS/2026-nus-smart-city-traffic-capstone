"""Stage 3: Matplotlib figures that reveal traffic patterns."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "traffic_features.csv"
FIGURE_DIR = BASE_DIR / "figures"


def _rel(path):
    """Show paths relative to this folder in log messages (shorter and portable)."""
    try:
        return Path(path).resolve().relative_to(BASE_DIR)
    except ValueError:
        return path
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Short interpretation written next to each figure (also saved to a text file)
INTERPRETATIONS = {
    "01_traffic_by_hour.png": (
        "Weekdays show two clear rush hours: a morning peak at 07:00 (about 6,000 vehicles/hour) and a "
        "slightly higher evening peak at 16:00 (about 6,200). Weekends have no sharp peaks; traffic "
        "builds gradually to about 4,400 at midday and fades in the evening. Overnight traffic is "
        "very low on both, and weekend nights are slightly busier than weekday nights."),
    "02_traffic_distribution.png": (
        "The distribution is not a single bell shape. There is a large spike of quiet overnight hours "
        "below about 1,000 vehicles, a middle group near 2,800 and a busy group around 4,500 to "
        "5,000. The quartile lines show where the Low / Medium / High / Severe congestion "
        "categories begin (1,248 / 3,427 / 4,952), so Severe means roughly the busiest quarter of hours."),
    "03_traffic_by_weather.png": (
        "Average traffic is highest in cloudy and hazy hours (about 3,600) and lowest in fog (about 2,700) and mist. "
        "Weather type is also tied to time of day (for example, fog is common at night), so this "
        "chart should not be read as weather causing the difference."),
    "04_temperature_vs_traffic.png": (
        "There is no strong straight-line link between temperature and traffic (correlation about "
        "0.14). Hours at every temperature range show the full spread from empty to very busy, so "
        "temperature alone is a weak predictor."),
    "05_congestion_heatmap.png": (
        "Heavy traffic is concentrated on weekdays from about 06:00 to 18:00, with the darkest blocks "
        "at 07:00 and 15:00 to 17:00. Saturday and Sunday are lighter throughout, and every day is "
        "quiet between midnight and 04:00."),
}


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    path = out_dir / name
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    except OSError:
        logger.error("Could not save figure %s", path, exc_info=True)
        raise
    finally:
        plt.close(fig)
    logger.info("Saved figure: %s", _rel(path))


def plot_traffic_by_hour(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for flag, label, colour in [(0, "Weekday", "#1f77b4"), (1, "Weekend", "#ff7f0e")]:
        means = df[df["is_weekend"] == flag].groupby("hour")["traffic_volume"].mean()
        ax.plot(means.index, means.values, marker="o", label=label, color=colour)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average traffic volume (vehicles/hour)")
    ax.set_title("Average traffic by hour: weekday vs weekend")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "01_traffic_by_hour.png", out_dir)


def plot_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["traffic_volume"], bins=50, color="#4c78a8", edgecolor="white")
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75])
    logger.debug("Distribution plot quartiles: %.1f %.1f %.1f", q1, q2, q3)
    for q, name in [(q1, "Q1"), (q2, "Median"), (q3, "Q3")]:
        ax.axvline(q, color="crimson", linestyle="--")
        ax.text(q, ax.get_ylim()[1] * 0.95, f" {name}\n {q:,.0f}", color="crimson", va="top")
    ax.set_xlabel("Traffic volume (vehicles/hour)")
    ax.set_ylabel("Number of hours")
    ax.set_title("Distribution of hourly traffic volume with quartile cut-offs")
    _save(fig, "02_traffic_distribution.png", out_dir)


def plot_by_weather(df: pd.DataFrame, out_dir: Path, min_hours: int = 100) -> None:
    stats = df.groupby("weather_main")["traffic_volume"].agg(["mean", "count"])
    small = stats[stats["count"] < min_hours]
    if len(small):
        logger.warning("Weather chart omits %d categories with fewer than %d hours: %s",
                       len(small), min_hours, ", ".join(small.index))
    stats = stats[stats["count"] >= min_hours].sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(stats.index, stats["mean"], color="#59a14f")
    for bar, n in zip(bars, stats["count"]):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2, f"n={n:,}", va="center", fontsize=8)
    overall = df["traffic_volume"].mean()
    ax.axvline(overall, color="grey", linestyle="--", label=f"Overall average ({overall:,.0f})")
    ax.set_xlabel("Average traffic volume (vehicles/hour)")
    ax.set_title("Average traffic by weather condition")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), frameon=False)
    _save(fig, "03_traffic_by_weather.png", out_dir)


def plot_temperature(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    hb = ax.hexbin(df["temp_c"], df["traffic_volume"], gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Number of hours")
    corr = df["temp_c"].corr(df["traffic_volume"])
    logger.debug("Temperature-traffic correlation: %.4f", corr)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Traffic volume (vehicles/hour)")
    ax.set_title(f"Temperature vs traffic volume (correlation = {corr:.2f})")
    _save(fig, "04_temperature_vs_traffic.png", out_dir)


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    grid = df.pivot_table(index="day_of_week", columns="hour", values="traffic_volume", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(grid.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_NAMES)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_title("Average traffic volume by day of week and hour")
    fig.colorbar(im, ax=ax, label="Vehicles/hour")
    _save(fig, "05_congestion_heatmap.png", out_dir)


def run_visualizations(in_path: Path = FEATURES_PATH, out_dir: Path = FIGURE_DIR) -> None:
    try:
        df = pd.read_csv(in_path, parse_dates=["date_time"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Feature file not found: {in_path}. Run main.py first.") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Visualisation stage started: %d rows loaded", len(df))
    for step in (plot_traffic_by_hour, plot_distribution, plot_by_weather, plot_temperature, plot_heatmap):
        step(df, out_dir)

    notes = ["# Figure interpretations\n"]
    for name, text in INTERPRETATIONS.items():
        notes.append(f"**{name}**: {text}\n")
    (out_dir / "figure_interpretations.md").write_text("\n".join(notes), encoding="utf-8")
    logger.info("Saved figure interpretations: %s", _rel(out_dir / "figure_interpretations.md"))
