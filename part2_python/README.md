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

**Text standardisation.** Leading and trailing spaces are trimmed, `weather_main` is title-cased and
`weather_description` is lower-cased. This changes the capitalisation of 1,730 raw rows ("Sky is Clear" becomes
"sky is clear", 1,726 rows, and "SQUALLS" becomes "squalls", 4 rows) and does not change which row is kept for
an hour or any number. The holiday text "None" is relabelled "No Holiday" (see above).

**Impossible values** are replaced with the median of valid values in the same calendar month, **pooled across all
years** (every January together, every February together and so on, not year by year). Each month is processed in a
loop:

| Problem | Rows | Rule |
| --- | --- | --- |
| Temperature of 0 K (-273 °C) | 10 | same-month median temperature |
| Rainfall of 9,831 mm in one hour | 1 | same-month median rainfall |
| Zero vehicles for a full hour (2016-07-23 at 18:00 and 23:00) | 2 | same-month **and same-hour** median traffic (pooled across all years), because traffic depends strongly on the hour |

In total 13 values are repaired. Temperature and rainfall follow the plain monthly-median rule. Traffic uses the
month-and-hour median (also pooled across all years), a deliberate refinement of that rule: a zero at 18:00 should be replaced by a typical
18:00 reading, not by a monthly figure that is mostly overnight and daytime hours. Rainfall comes out as 0.0 mm
because July's median hourly rainfall is zero (most hours are dry).

**Known anomaly: 22-24 July 2016.** The two zero-traffic hours sit inside a longer stretch that looks like a
sensor or road outage. From 22:00 on 22 July to about 17:00 on 24 July (44 hours), hourly counts are mostly tiny:
28 of the 44 hours are under 100 vehicles, against 7 of about 4,200 July hours outside the stretch. A typical July
22:00 reading is about 2,200, but this stretch starts with 64 and 10 vehicles. Readings partly recover in the early
mornings of 23 and 24 July (05:00 to 08:00), then collapse again. The two zeros were repaired as the rule above
requires, but the repaired value at 18:00 (4,279) is far above its neighbours (5 vehicles at 17:00 and 1 at 19:00), so
the repaired values should be read as placeholders. The other hours in the stretch are left unchanged and no rows are
removed in Part 2. This stretch is suspect data: Part 3 should exclude or flag the whole window from 22 July 22:00 to
24 July 17:00 and say so in its own documentation. It still needs checking with the data owner.

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

## Feature dictionary

`traffic_features.csv` has 43 columns: the 9 cleaned columns plus 34 engineered ones.

| Column(s) | Definition |
| --- | --- |
| `hour`, `day_of_week`, `month`, `year` | Parts of `date_time`. Day of week: Monday = 0 to Sunday = 6 |
| `is_weekend` | 1 for Saturday and Sunday, otherwise 0 (Friday is a weekday) |
| `is_rush_hour` | 1 on weekdays at 07:00, 08:00, 16:00, 17:00 or 18:00 (peaks seen in Part 1) |
| `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | Position on a circle for hour (24), day of week (7) and month (12), so 23:00 sits next to 00:00 and December next to January |
| `is_holiday` | 1 if the holiday name is not "No Holiday" (filled across the whole calendar day) |
| `temp_c` | Temperature in °C (`temp` in kelvin minus 273.15) |
| `is_raining` | 1 if `rain_1h` is above 0 mm |
| `is_snowing` | 1 if `snow_1h` is above 0 mm |
| `is_low_visibility` | 1 if `weather_main` is Fog, Mist, Haze or Smoke |
| `is_precipitation` | 1 if `weather_main` is Rain, Drizzle, Snow, Thunderstorm or Squall |
| `is_severe_weather` | 1 if `weather_main` is Thunderstorm, Snow or Squall |
| `weather_*` (11 columns) | One-hot encoding of `weather_main`, one 0/1 column per category |
| `temp_c_scaled` | `temp_c` standardised (z-score: minus the mean, divided by the sample standard deviation, n - 1) |
| `clouds_scaled` | `clouds_all` rescaled to 0-1 (minimum to maximum) |
| `rain_scaled` | `log(1 + rain_1h)` rescaled to 0-1, because rainfall is very skewed |
| `congestion_category` | Low, Medium, High or Severe from the traffic-volume quartiles (see Data decisions) |

The weather groups are defined once in `config.py`. `is_low_visibility` and `is_precipitation` use the
weather category of the chosen hourly row, while `is_raining` and `is_snowing` use the measured amounts. The
two can disagree: 6,627 hours are labelled Rain, Snow, Thunderstorm, Drizzle or Squall but record 0 mm of rain
(for example, a snow hour). `is_precipitation` therefore means "the weather label says precipitation", and
`is_raining` means "rain was actually measured".

### Scaling and data leakage (note for Part 3)

The scaled columns and the quartile cut-offs in `congestion_category` are calculated on **all years**,
including 2018. That suits Part 2, which describes the whole dataset and reconciles with the Part 1 numbers.
It is not suitable for model evaluation, because Part 3 trains on 2017 and earlier and tests on 2018, and the
test year should not influence how the training data is scaled. Part 3 therefore **refits any scaler on the
training years only** and applies those fixed values to the test year. The Part 2 scaled columns and the
quartile category are descriptive and should not be used to fit or evaluate a model unless recalculated on the
training years.

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
