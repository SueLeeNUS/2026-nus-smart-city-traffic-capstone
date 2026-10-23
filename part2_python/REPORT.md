---
title: "Part 2 Report: Reproducible Traffic Analytics Pipeline"
---

# Part 2 Report: Reproducible Traffic Analytics Pipeline

Sue Lee · Smart City Traffic Intelligence capstone · I-94 westbound, Minneapolis–St Paul

## 1. Purpose

Part 1 found that the raw file has data-quality problems and that traffic follows the clock and calendar far more
than the weather. Part 2 turns that understanding into a repeatable Python workflow: clean the data, create
features for the Part 3 models, visualise the patterns and let a user query the results. Every step writes to a log
so the processing can be audited, and every data rule matches the Part 1 SQL view so the two parts report the same numbers.

## 2. Methodology

**Cleaning.** The raw file has 48,204 rows and 9 columns. The pipeline checks the schema first, then:

- Relabels the holiday text `"None"` as `"No Holiday"`. It is a real value, not a missing one.
- Drops 17 exact duplicate rows.
- **Collapses to one row per hour.** A further 7,612 rows repeat an hour already recorded. They carry extra weather
  labels but identical traffic, so we keep the row with the most severe weather (Thunderstorm > Squall > Snow >
  Rain > Drizzle > Fog > Mist > Haze > Smoke > Clouds > Clear). This leaves **40,575 hourly records**, and a check
  confirms every timestamp is unique.
- **Fills the holiday flag across the whole day.** Only 00:00 was flagged (53 hours), so the flag carried almost no signal. It
  now covers **1,203 hours across 53 holiday dates**.
- Replaces 10 impossible 0 K temperatures and one 9,831 mm rainfall reading with the median of valid readings in
  the same calendar month, and 2 zero-traffic hours (23 July 2016, 18:00 and 23:00, sensor dropouts) with the median
  for the same month and hour. Each is calculated group by group in a loop, not with one global average.

Each step writes its own log line with the number of rows affected and the reason.

**Reconciliation with Part 1.** Compared hour by hour with the SQL view, the Python output has the same 40,575
hours, the same weather label and holiday label for every hour, and the same 6,107 hours above 5,500 vehicles. The
only differences are the 13 repaired values, which move the mean from 3,290.65 to 3,290.79.

**Features.** The cleaned data gains 34 columns (43 in total): hour, day of week, month, weekend and rush-hour
flags, sine/cosine pairs for hour, day of week and month (so 23:00 sits next to 00:00), a holiday flag, temperature
in °C, rain, snow, low-visibility, precipitation and severe-weather flags, one-hot weather categories, and scaled
versions of temperature (z-score), cloud cover and rainfall (0–1).

**Three congestion definitions, kept distinct.** Policy-facing: volume above 5,500 (15.1% of hours). Dashboard-facing:
bands at 4,500 and 5,500. Modelling construct: quartiles of volume (1,249 / 3,428 / 4,952) giving Low, Medium,
High and Severe with about a quarter of the hours each. The cut-offs come from the data, so no hand-picked
number is needed for the target.

## 3. Findings from the figures

1. **Hourly pattern.** Weekdays have two rush hours, about 6,000 vehicles/hour at 07:00 and 6,200 at 16:00.
   Weekends have no sharp peak and rise gradually to about 4,400 at midday.
2. **Distribution.** Hourly traffic is not bell-shaped. A large group of quiet overnight hours (under about 1,000)
   sits apart from busy daytime hours (4,500–5,000), which is why quartiles work well for the categories.
3. **Weather.** Cloudy and hazy hours average 3,600–3,700 vehicles/hour, fog about 2,650. Weather type is tied to
   time of day, so these gaps are not evidence that weather causes the difference.
4. **Temperature.** Correlation with traffic is about 0.14, a weak link that agrees with Part 1.
5. **Heatmap.** Heavy traffic sits on weekdays between 06:00 and 18:00, peaking at 07:00 and 15:00–17:00.

The app confirms the weekday–weekend gap: weekdays average 3,557 vehicles/hour and 33.8% of weekday hours are
Severe, against 2,624 and 3.1% on weekends.

## 4. Mini application

`cli_app/traffic_app.py` offers five commands (`at`, `peaks`, `compare`, `recommend`, `weather`). It logs each
command with its arguments and reports invalid input, such as a malformed date, as a clear ERROR without a traceback.

## 5. Limitations and next steps

- About nine months are missing (September 2014 to May 2015), so a query for a missing hour returns a clear message
  and yearly totals reflect coverage, not demand.
- Keeping one weather label per hour discards the less severe labels. This favours bad-weather hours and is stated
  so Part 3 can account for it.
- Smoke (2 hours) and squall (1 hour) are too rare for reliable averages; the weather chart leaves them out.
- The 23 July 2016 anomaly (very low traffic all day) remains in the data apart from the two zero readings and still
  needs checking with the data owner.
- Part 3 uses `traffic_clean.csv` and `traffic_features.csv`, the shared weather groups and the seed in `config.py`.
