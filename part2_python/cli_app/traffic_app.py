"""Mini Traffic Analytics Application (command line).

Run `python main.py` (in the part2_python folder) first so the processed data exists.
Then, from the part2_python folder, for example:

    python cli_app/traffic_app.py at "2017-05-12 16:00"
    python cli_app/traffic_app.py peaks --top 5 --day-type weekday
    python cli_app/traffic_app.py compare
    python cli_app/traffic_app.py recommend --day Fri
    python cli_app/traffic_app.py weather --condition Rain
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# This file lives in cli_app/, so add the parent folder to find logging_config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = APP_DIR.parent / "data" / "processed" / "traffic_features.csv"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class UserInputError(Exception):
    """The user supplied something we cannot use (bad date, unknown day...)."""


def load_data(path: Path = FEATURES_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
    except FileNotFoundError as exc:
        raise UserInputError(f"Processed data not found at {path}. Run 'python main.py' first.") from exc
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


# ---------------------------------------------------------------- commands
def cmd_at(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Traffic and weather at a specific date and hour."""
    try:
        when = pd.to_datetime(args.datetime, format="%Y-%m-%d %H:%M", errors="raise").floor("h")
    except (ValueError, TypeError) as exc:
        raise UserInputError(
            f"'{args.datetime}' is not a valid date/time. Use the format YYYY-MM-DD HH:MM, "
            "for example 2017-05-12 16:00.") from exc
    row = df[df["date_time"] == when]
    if row.empty:
        raise UserInputError(
            f"No record for {when}. Data covers {df['date_time'].min()} to {df['date_time'].max()} "
            "and has gaps (for example Aug 2014 to Jun 2015).")
    r = row.iloc[0]
    print(f"Time:        {r['date_time']}  ({DAY_NAMES[int(r['day_of_week'])]})")
    print(f"Traffic:     {int(r['traffic_volume']):,} vehicles/hour  ({r['congestion_category']} congestion)")
    print(f"Weather:     {r['weather_main']} ({r['weather_description']})")
    print(f"Temperature: {r['temp_c']:.1f} C   Rain: {r['rain_1h']} mm   Snow: {r['snow_1h']} mm   Clouds: {int(r['clouds_all'])}%")
    print(f"Holiday:     {r['holiday']}")


def cmd_peaks(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Hours of the day with the highest average traffic."""
    if args.top < 1 or args.top > 24:
        raise UserInputError("--top must be between 1 and 24.")
    data = df
    if args.day_type == "weekday":
        data = df[df["is_weekend"] == 0]
    elif args.day_type == "weekend":
        data = df[df["is_weekend"] == 1]
    means = data.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False).head(args.top)
    print(f"Top {args.top} busiest hours ({args.day_type} average):")
    for hour, value in means.items():
        print(f"  {int(hour):02d}:00  {value:,.0f} vehicles/hour")


def cmd_compare(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Weekday versus weekend traffic."""
    print(f"{'':10s}{'Average':>10s}{'Median':>10s}{'Max':>8s}{'% Severe hours':>16s}")
    for label, flag in (("Weekday", 0), ("Weekend", 1)):
        d = df[df["is_weekend"] == flag]
        severe = (d["congestion_category"] == "Severe").mean() * 100
        print(f"{label:10s}{d['traffic_volume'].mean():>10,.0f}{d['traffic_volume'].median():>10,.0f}"
              f"{d['traffic_volume'].max():>8,.0f}{severe:>15.1f}%")


def cmd_recommend(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Quietest daytime hours (06:00-21:59) for a given day of the week."""
    day = args.day.strip().capitalize()[:3]
    if day not in DAY_NAMES:
        raise UserInputError(f"'{args.day}' is not a day. Use one of: {', '.join(DAY_NAMES)}.")
    d = df[(df["day_of_week"] == DAY_NAMES.index(day)) & df["hour"].between(6, 21)]
    means = d.groupby("hour")["traffic_volume"].mean().sort_values().head(args.top)
    print(f"Recommended travel hours on {day} (lowest average traffic, 06:00-21:59):")
    for hour, value in means.items():
        print(f"  {int(hour):02d}:00  {value:,.0f} vehicles/hour")
    busiest = d.groupby("hour")["traffic_volume"].mean().idxmax()
    print(f"Avoid around {int(busiest):02d}:00, the busiest hour on {day}.")


def cmd_weather(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Average traffic under a given weather condition."""
    options = sorted(df["weather_main"].unique())
    match = [w for w in options if w.lower() == args.condition.strip().lower()]
    if not match:
        raise UserInputError(f"'{args.condition}' is not a known weather condition. Options: {', '.join(options)}.")
    d = df[df["weather_main"] == match[0]]
    print(f"{match[0]}: {len(d):,} hours recorded")
    print(f"  Average traffic: {d['traffic_volume'].mean():,.0f} vehicles/hour "
          f"(overall average {df['traffic_volume'].mean():,.0f})")
    print(f"  Share of Severe-congestion hours: {(d['congestion_category'] == 'Severe').mean() * 100:.1f}%")
    if len(d) < 100:
        print("  Note: fewer than 100 hours, so treat this average with caution.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the processed I-94 traffic data.")
    parser.add_argument("--debug", action="store_true", help="show DEBUG messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("at", help="traffic and weather at a date/time")
    p.add_argument("datetime", help='e.g. "2017-05-12 16:00"')
    p.set_defaults(func=cmd_at)

    p = sub.add_parser("peaks", help="busiest hours of the day")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--day-type", default="all", choices=["all", "weekday", "weekend"])
    p.set_defaults(func=cmd_peaks)

    p = sub.add_parser("compare", help="weekday vs weekend traffic")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("recommend", help="quietest travel hours for a day of the week")
    p.add_argument("--day", required=True, help="Mon, Tue, ... Sun")
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("weather", help="traffic under a weather condition")
    p.add_argument("--condition", required=True, help="e.g. Rain, Snow, Fog")
    p.set_defaults(func=cmd_weather)
    return parser


def main(argv=None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed its own usage message
        return int(exc.code or 0)
    # Console shows only warnings/errors so the printed answers stay clean;
    # the log file records INFO (commands invoked) and above, appended each run.
    setup_logging(str(APP_DIR / "app.log"), debug=args.debug, console_level=logging.WARNING, file_mode="a")
    arguments = {k: v for k, v in vars(args).items() if k not in ("func", "debug")}
    logger.info("Command invoked: %s with arguments %s", args.command, arguments)
    try:
        df = load_data()
        args.func(df, args)
    except UserInputError as exc:
        logger.error("Invalid input: %s", exc)
        return 2
    except Exception:  # unexpected problem: log details, exit without a raw traceback
        logger.error("Unexpected error while running '%s'", args.command, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    # Re-import under the name "traffic_app" so log lines show a clear module name
    from traffic_app import main as _main

    sys.exit(_main())
