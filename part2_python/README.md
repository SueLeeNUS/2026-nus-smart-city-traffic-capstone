# Capstone Part 2: Traffic Analytics Pipeline (I-94 Westbound, Minneapolis–St Paul)

A reproducible Python workflow that cleans the raw hourly traffic CSV, engineers machine-learning-ready
features, draws figures, and offers a small command-line tool for querying the processed data.
It builds on the Part 1 findings (duplicate timestamps, 0 K temperatures, the 9,831 mm rainfall outlier and
the `"None"` holiday code).

## Project structure

This folder (`part2_python/`) sits inside the capstone repository next to `part1_data_analytics/` and
`part3_machine_learning/`.

```
part2_python/
├── main.py                  # runs the whole project: clean -> features -> figures
├── pipeline.py              # Task 1: load, validate schema, clean (also runnable alone)
├── feature_engineering.py   # Task 2: time, weather, scaled features, congestion category
├── visualizations.py        # Task 3: five Matplotlib figures + interpretations
├── logging_config.py        # shared logging setup (called only by entry-point scripts)
├── cli_app/
│   ├── traffic_app.py       # Task 4: command-line analytics app
│   └── app.log              # sample log from the app
├── figures/                 # five PNG charts + figure_interpretations.md
├── data/
│   ├── raw/Metro_Interstate_Traffic_Volume.csv
│   └── processed/           # traffic_clean.csv, traffic_features.csv (created by main.py)
├── pipeline.log             # sample log from a full end-to-end run
├── requirements.txt
├── REPORT.md / REPORT.docx  # 1-2 page methodology and findings report
└── README.md
```

## How to run

Requires Python 3.10+.

```bash
cd part2_python
pip install -r requirements.txt

python main.py             # full pipeline: clean -> features -> figures (INFO and above)
python main.py --debug     # same, but also shows DEBUG messages
python pipeline.py         # cleaning stage only
```

Then query the processed data (run `main.py` first, from inside `part2_python/`):

```bash
python cli_app/traffic_app.py at "2017-05-12 16:00"            # traffic + weather at a date/time
python cli_app/traffic_app.py peaks --top 5 --day-type weekday # busiest hours of the day
python cli_app/traffic_app.py compare                          # weekday vs weekend
python cli_app/traffic_app.py recommend --day Fri              # quietest travel hours for a day
python cli_app/traffic_app.py weather --condition Rain         # traffic under a weather condition
```

Invalid input (for example a malformed date) is logged as an ERROR with a clear message, and the program
exits without a traceback.

## What the pipeline does

**Task 1 – Cleaning (`pipeline.py`)**: loads the CSV with `try/except` handling, checks that all 9 expected
columns exist *before* anything else, then runs separate cleaning steps:

| Step | Rule |
|---|---|
| Categoricals | trim whitespace, standard capitalisation; holiday `"None"` relabelled `"No Holiday"` (a real value, not missing) |
| Dates | `date_time` parsed; unparseable rows would be dropped |
| Exact duplicates | 17 identical rows dropped |
| Repeated timestamps | 7,612 further rows dropped: the same hour listed several times with extra weather labels but identical traffic. First row per hour kept, leaving 40,575 unique hours |
| Outliers | 10 temperatures of 0 K and 1 rainfall of 9,831 mm replaced with the **median of the same calendar month** (processed month by month in a loop) |

**Task 2 – Features (`feature_engineering.py`)**: hour, day of week, month, weekend flag, rush-hour flag,
sine/cosine encodings of hour, day of week and month, holiday flag, °C temperature, rain/snow flags,
low-visibility / precipitation / severe-weather flags, one-hot weather categories, scaled temperature
(z-score), scaled cloud cover and log-scaled rainfall (min-max), and the target `congestion_category`.

*Congestion logic*: take the 25th, 50th and 75th percentiles of `traffic_volume` (1,248 / 3,427 / 4,952).
Volume ≤ Q1 is **Low**, ≤ Q2 **Medium**, ≤ Q3 **High**, above Q3 **Severe**. Each category holds about a quarter
of the hours. (Part 1's Power BI dashboard used a fixed 5,500 threshold; Part 2 uses quartiles as the capstone
instructions specify.)

**Task 3 – Figures (`visualizations.py`)**: hourly pattern (weekday vs weekend), traffic distribution with
quartiles, traffic by weather, temperature vs traffic, and a day-of-week × hour heatmap. Interpretations are in
`figures/figure_interpretations.md` and in the report.

**Task 4 – App (`cli_app/traffic_app.py`)**: the five commands above.

## Logging configuration

* Every module uses `logger = logging.getLogger(__name__)`; no module uses the root logger directly.
* Handlers are configured **only in entry-point scripts** (`main.py`, `pipeline.py` when run directly and
  `cli_app/traffic_app.py`) through `logging_config.setup_logging()`, which writes to **both the console and a log file**.
* Format: `timestamp | LEVEL | module | message`.
* Log files: `pipeline.log` (pipeline, overwritten each run) and `cli_app/app.log` (command-line app, appended).
  The app shows only warnings/errors on the console so its printed answers stay readable.
* `print()` is used only in `cli_app/traffic_app.py` for the answers the user asked for, never for progress messages.

| Level | Meaning in this project |
|---|---|
| DEBUG | Fine-grained values (quartile thresholds, scaling means, monthly median replacements). Shown only with `--debug` |
| INFO | Normal milestones: file loaded (rows/columns), schema check passed, step completed, figure saved, command invoked |
| WARNING | Unexpected but recoverable: rows dropped, values imputed, categories omitted from a chart |
| ERROR | Something prevents continuing: missing file, schema failure, invalid user input. Pipeline errors include the traceback (`exc_info=True`) |

## Git history

One commit per task (cleaning, features, figures, app, then documentation) so the progression of the work is visible
with `git log --oneline`.
