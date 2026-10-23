# Capstone Part 2: Traffic Analytics Pipeline (I-94 Westbound, Minneapolis–St Paul)

A reproducible Python workflow that cleans the raw hourly traffic CSV, engineers machine-learning-ready
features, draws figures, and offers a small command-line tool for querying the processed data.
It builds on the Part 1 findings and uses the same data rules as the Part 1 SQL view `traffic_hourly`, so
SQL and Python report the same numbers.

## Project structure

This folder (`part2_python/`) sits inside the capstone repository next to `part1_data_analytics/` and
`part3_machine_learning/`.

```
part2_python/
├── main.py                  # runs the whole project: clean -> features -> figures
├── pipeline.py              # Task 1: load, validate schema, clean (also runnable alone)
├── feature_engineering.py   # Task 2: time, weather, scaled features, congestion category
├── visualizations.py        # Task 3: five Matplotlib figures + interpretations
├── config.py                # shared decisions: grain rule, severity order, weather groups, seed
├── logging_config.py        # shared logging setup (called only by entry-point scripts)
├── cli_app/
│   ├── traffic_app.py       # Task 4: command-line analytics app
│   └── app.log              # sample log from the app
├── figures/                 # five PNG charts + figure_interpretations.md
├── data/
│   ├── raw/Metro_Interstate_Traffic_Volume.csv
│   └── processed/           # traffic_clean.csv (one row per hour), traffic_features.csv (created by main.py)
├── pipeline.log             # sample log: a full DEBUG run, then a deliberate failing run (ERROR)
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

## Data decisions

These decisions are made once, in `config.py` and `pipeline.py`, and reused in Part 3.

**Grain: one row per hour.** The raw file (48,204 rows) lists the same hour once per weather description, always
with identical traffic volume. After removing 17 exact duplicates, 7,612 further rows are repeated hours. We keep
**one row per hour: the one with the most severe `weather_main`**, ties broken alphabetically by
`weather_description`. Result: **40,575 hourly rows**, checked for unique timestamps straight after collapsing.

Severity order, most severe first: **Thunderstorm > Squall > Snow > Rain > Drizzle > Fog > Mist > Haze > Smoke >
Clouds > Clear.**

**Holiday flag: filled across the whole day.** In the raw file only the 00:00 row of a holiday carries its name
(53 flagged hours), which makes the flag nearly useless. The pipeline copies each holiday name to every hour of
that calendar date: **1,203 holiday hours across 53 dates**. The change is logged as a WARNING with before and
after counts.

**Impossible values** are replaced with the median of valid values in the same calendar month (processed month
by month in a loop):

| Problem | Rows | Rule |
| --- | --- | --- |
| Temperature of 0 K (-273 °C) | 10 | same-month median temperature |
| Rainfall of 9,831 mm in one hour | 1 | same-month median rainfall |
| Zero vehicles for a full hour (2016-07-23 at 18:00 and 23:00) | 2 | same-month **and same-hour** median traffic, because traffic depends strongly on the hour |

**Three congestion definitions are kept separate on purpose:**

1. **Fixed threshold, policy-facing (Part 1 probability tasks):** traffic above 5,500 vehicles/hour, which is 6,107
   of 40,575 hours (15.1%).
2. **Bands, dashboard-facing (Part 1 Power BI):** Low below 4,500, Medium 4,500 to 5,500, High above 5,500.
3. **Quartiles, a modelling construct (Parts 2 and 3):** cut-offs computed from the data. On the cleaned data
   they are **1,249 / 3,428 / 4,952** (Low, Medium, High, Severe). The Part 1 plan quotes 1,248 / 3,427 / 4,952;
   the difference of one vehicle comes from the two repaired zero readings above.

**Proxy-label weather groups** (reused in Part 3): `SEVERE_WEATHER` = Thunderstorm, Snow, Squall;
`LOW_VISIBILITY_WEATHER` = Fog, Mist, Haze, Smoke. The random seed constant (`RANDOM_SEED = 42`) also lives in
`config.py`.

## Reconciliation with Part 1 (SQL)

The cleaned Python output was compared hour by hour with the Part 1 SQLite view `traffic_hourly`:

| Check | SQL view | Python output |
| --- | --- | --- |
| Hourly rows | 40,575 | 40,575 |
| Weather label per hour | identical | identical |
| Holiday hours / dates | 1,203 / 53 | 1,203 / 53 |
| Hours above 5,500 vehicles | 6,107 | 6,107 |
| Mean traffic per hour | 3,290.65 | 3,290.79 |

The only differences are the 13 values the pipeline deliberately repairs (10 temperatures, 1 rainfall, 2 traffic
readings). They explain the small gap in the mean.

## What the pipeline does

**Task 1 – Cleaning (`pipeline.py`)**: loads the CSV with `try/except` handling for a missing, empty, unreadable or
malformed file, checks that all 9 expected columns exist *before* anything else, then runs separate cleaning
steps, each with its own log message: standardise categoricals, parse dates, check numbers, drop exact
duplicates, collapse to one row per hour, verify uniqueness, fill the holiday flag, impute impossible values.

**Task 2 – Features (`feature_engineering.py`)**: hour, day of week, month, weekend flag, rush-hour flag,
sine/cosine encodings of hour, day of week and month, holiday flag, °C temperature, rain/snow flags,
low-visibility / precipitation / severe-weather flags, one-hot weather categories, scaled temperature (z-score),
scaled cloud cover and log-scaled rainfall (min-max), and the target `congestion_category`.

*Quartile logic*: Volume ≤ Q1 is **Low**, ≤ Q2 **Medium**, ≤ Q3 **High**, above Q3 **Severe**. Each category holds
about a quarter of the hours.

**Task 3 – Figures (`visualizations.py`)**: hourly pattern (weekday vs weekend), traffic distribution with
quartiles, traffic by weather, temperature vs traffic, and a day-of-week × hour heatmap. Interpretations are in
`figures/figure_interpretations.md` and in the report.

**Task 4 – App (`cli_app/traffic_app.py`)**: the five commands above.

## Logging configuration

* Every module uses `logger = logging.getLogger(__name__)`; no module uses the root logger directly.
* Handlers are configured **only in entry-point scripts** (`main.py`, `pipeline.py` when run directly and
  `cli_app/traffic_app.py`) through `logging_config.setup_logging()`, which writes to **both the console and a log file**.
* Format: `timestamp | LEVEL | module | message`.
* Log files: `pipeline.log` (pipeline, replaced on each run) and `cli_app/app.log` (command-line app, appended).
  The app shows only warnings/errors on the console so its printed answers stay readable.
* `print()` is used only in `cli_app/traffic_app.py` for the answers the user asked for, never for progress messages.

| Level | Meaning in this project |
|---|---|
| DEBUG | Fine-grained values (quartile cut-offs, scaling means, median replacements). Shown only with `--debug` |
| INFO | Normal milestones: file loaded (rows/columns), schema check passed, step completed, figure saved, command invoked |
| WARNING | Unexpected but recoverable: rows dropped, values imputed or modified, categories omitted from a chart |
| ERROR | Something prevents continuing: missing file, schema failure, invalid user input. Pipeline errors include the traceback (`exc_info=True`) |

**How the sample `pipeline.log` was produced:** it holds two runs, so all four levels are visible without re-running
anything:

```bash
python main.py --debug                                                              # DEBUG, INFO, WARNING
python main.py --debug --raw-path data/raw/does_not_exist.csv --append-log          # ERROR with traceback
```

## Git history

One commit per task (cleaning, features, figures, app, then documentation) so the progression of the work is visible
with `git log --oneline`.
