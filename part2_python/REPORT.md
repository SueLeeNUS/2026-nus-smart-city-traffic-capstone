---
title: "Part 2 Report: Reproducible Traffic Analytics Pipeline"
---

# Part 2 Report: Reproducible Traffic Analytics Pipeline

Sue Lee · Smart City Traffic Intelligence capstone · I-94 westbound, Minneapolis–St Paul

## 1. Purpose

Part 1 found that the raw file has data-quality problems and that traffic follows the clock and calendar far more
than the weather. Part 2 turns that understanding into a repeatable Python workflow: clean the data, create
features for the Part 3 models, visualise the patterns and let a user query the results. Every step writes to a log
so the processing can be audited.

## 2. Methodology

**Cleaning.** The raw file has 48,204 rows and 9 columns. The pipeline checks the schema first, then:

- Relabels the holiday text `"None"` as `"No Holiday"`. It is a real value, not a missing one (a Part 1 finding).
- Drops 17 exact duplicate rows.
- Drops 7,612 rows that repeat an hour already recorded. These carry extra weather labels but the same traffic
  count, so keeping the first row per hour loses no traffic information. This leaves **40,575 hourly records**,
  matching the 40,575 distinct timestamps found in Part 1.
- Replaces 10 impossible 0 K temperatures and one 9,831 mm rainfall reading with the median of valid readings in
  the same calendar month, calculated month by month rather than with one global average.

Each step writes its own log line with the number of rows affected and the reason.

**Features.** The cleaned data gains 34 columns (43 in total): hour, day of week, month, weekend and rush-hour
flags, sine/cosine pairs for hour, day of week and month (so 23:00 sits next to 00:00), a holiday flag, temperature
in °C, rain, snow, low-visibility, precipitation and severe-weather flags, one-hot weather categories, and scaled
versions of temperature (z-score), cloud cover and rainfall (0–1).

**Congestion category.** The 25th, 50th and 75th percentiles of traffic volume (1,248 / 3,427 / 4,952 vehicles
per hour) split the hours into Low, Medium, High and Severe, each holding about a quarter of the data. The
thresholds come from the data, so no hand-picked cut-off is needed. This differs from the fixed 5,500 line used on the
Part 1 dashboard, because the capstone instructions ask for quartiles here.

## 3. Findings from the figures

1. **Hourly pattern.** Weekdays have two rush hours, about 6,000 vehicles/hour at 07:00 and 6,200 at 16:00.
   Weekends have no sharp peak and rise gradually to about 4,400 at midday.
2. **Distribution.** Hourly traffic is not bell-shaped. A large group of quiet overnight hours (under about 1,000)
   sits apart from busy daytime hours (4,500–5,000), which is why quartiles work well for the categories.
3. **Weather.** Cloudy and hazy hours average about 3,600 vehicles/hour, fog about 2,700. Weather type is tied to
   time of day, so these gaps are not evidence that weather causes the difference.
4. **Temperature.** Correlation with traffic is about 0.14 on the cleaned data, a weak link that agrees with Part 1.
5. **Heatmap.** Heavy traffic sits on weekdays between 06:00 and 18:00, peaking at 07:00 and 15:00–17:00.

The app confirms the weekday–weekend gap: weekdays average 3,557 vehicles/hour and 33.8% of weekday hours are
Severe, against 2,624 and 3.1% on weekends.

## 4. Mini application

`cli_app/traffic_app.py` offers five commands (`at`, `peaks`, `compare`, `recommend`, `weather`). It logs each command with
its arguments and reports invalid input, such as a malformed date, as a clear ERROR without a traceback.

## 5. Limitations and next steps

- The data has gaps (for example August 2014 to June 2015), so a query for a missing hour returns a clear message.
- Keeping the first row per repeated hour is a simplification. The weather label of the other rows is discarded.
- Smoke (15 hours) and squall (1 hour) are too rare for reliable averages; the weather chart leaves them out.
- The 23 July 2016 anomaly flagged in Part 1 is kept in the data and still needs checking with the data owner.
- Part 3 will use `traffic_features.csv`, including the congestion category and the low-visibility and
  severe-weather flags, for the models.
