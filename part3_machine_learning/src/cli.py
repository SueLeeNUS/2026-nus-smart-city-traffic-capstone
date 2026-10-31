"""Command-line application. Part 3 promotes the Part 2 `recommend` command into the recommender.

Example:
    python -m src.cli recommend --datetime "2018-03-14 16:00:00" --temp 280 --weather Clear --window 3

Status messages go through logging. Only the answer is printed.
"""
import argparse
import logging
import sys

from src import config
from src.serving import load_classifier, load_regressor, payload_to_frame, predict_high_risk, predict_volume

logger = logging.getLogger("src.cli")


def configure_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(config.LOG_DIR / "cli.log", encoding="utf-8")],
    )


def recommend(args) -> int:
    logger.info("recommend invoked with %s", vars(args))
    base = {"datetime": args.datetime, "temp": args.temp, "weather_main": args.weather,
            "rain_1h": args.rain, "clouds_all": args.clouds, "holiday": args.holiday}
    try:
        payload_to_frame(base)
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        print(f"Sorry, I could not use that input: {exc}. Try: --datetime \"2018-03-14 16:00:00\"")
        return 2
    import pandas as pd
    requested = pd.to_datetime(args.datetime)
    reg = load_regressor()
    clf = load_classifier()
    rows = []
    for offset in range(-args.window, args.window + 1):
        when = requested + pd.Timedelta(hours=offset)
        p = {**base, "datetime": when.strftime("%Y-%m-%d %H:%M:%S")}
        out = predict_volume(reg, p)
        rows.append({"time": when.strftime("%H:%M"), "volume": out["predicted_volume"],
                     "band": out["congestion_band"], "high_risk_prob": predict_high_risk(clf, p)})
    best = min(rows, key=lambda r: r["volume"])
    here = next(r for r in rows if r["time"] == requested.strftime("%H:%M"))
    print(f"Predicted traffic around {requested:%Y-%m-%d %H:%M} (weather: {args.weather}, {args.temp} K):")
    print("  Volume uses the weather label. The high-risk probability uses temperature, rain, cloud cover")
    print("  and time only, because the weather label defines that proxy label (see the README).")
    for r in rows:
        print(f"  {r['time']}  ~{r['volume']:>5} veh/h  {r['band']:<6}  high-risk probability {r['high_risk_prob']:.0%}")
    if best["time"] == here["time"] or here["volume"] - best["volume"] < 0.10 * here["volume"]:
        print(f"Recommendation: {here['time']} is already close to the quietest option in this window. Keep the plan.")
    else:
        saving = here["volume"] - best["volume"]
        print(f"Recommendation: leave at {best['time']} instead. Expected traffic falls by about {saving} veh/h "
              f"({best['volume']} against {here['volume']}).")
    logger.info("recommend answered: best=%s requested=%s", best["time"], here["time"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic", description="Smart City Traffic command-line tools")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend", help="Suggest the quietest departure time near a requested time")
    rec.add_argument("--datetime", required=True, help='Requested time, e.g. "2018-03-14 16:00:00"')
    rec.add_argument("--temp", type=float, default=280.0, help="Air temperature in kelvin")
    rec.add_argument("--weather", default="Clear", help="weather_main label, e.g. Clear, Clouds, Rain, Snow")
    rec.add_argument("--rain", type=float, default=0.0, help="Rain in the last hour (mm)")
    rec.add_argument("--clouds", type=int, default=40, help="Cloud cover percent")
    rec.add_argument("--holiday", default="No Holiday", help="Holiday name, or No Holiday")
    rec.add_argument("--window", type=int, default=3, help="Hours either side of the requested time to check")
    rec.set_defaults(func=recommend)
    return parser


def main(argv=None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s. Run `python run_all.py` first.", exc)
        print("The models have not been trained yet. Run `python run_all.py` once, then try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
