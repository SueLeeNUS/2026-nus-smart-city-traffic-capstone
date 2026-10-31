# Capstone Part 1: Data Analytics (SQL, Statistics, Probability, Power BI)

Smart City Traffic Intelligence, I-94 westbound (Minneapolis-St Paul), hourly traffic volume with weather,
October 2012 to September 2018. This folder holds the Part 1 analysis: SQL queries on a SQLite database, a
statistical and probability analysis, a Power BI dashboard and a short insights report for a mobility team.

## What is in this folder

```
part1_data_analytics/
├── Part 1 - Data Analytics Insights Report.pdf       # 3 pages: summary, findings, data quality, dashboard picture
├── sql/
│   ├── part1_queries.sql                             # every SQL query used, in task order
│   └── Task1.db                                      # SQLite database: raw table `traffic` (48,204 rows)
├── powerbi/
│   ├── Task4_Dashboard_v2.pbix                       # Power BI dashboard (open with Power BI Desktop)
│   └── dashboard_screenshot.png                      # picture of the dashboard with all slicers cleared
└── statistics/
    └── Part 1 - Statistical and Probability Analysis.pdf   # 8 pages: Tasks 1 to 4 with working
```

GitHub cannot always preview the PDFs. Use the download button on the file page to open them.

## How to read the work

| If you want | Open |
| --- | --- |
| The short story for decision makers | `Part 1 - Data Analytics Insights Report.pdf` |
| Every figure with its method and interpretation | `statistics/Part 1 - Statistical and Probability Analysis.pdf` |
| The exact SQL behind each figure | `sql/part1_queries.sql` |
| The dashboard | `powerbi/dashboard_screenshot.png`, or the `.pbix` in Power BI Desktop |

## The one data decision that changes every number

The raw file has 48,204 rows but only **40,575 distinct hours**. The same hour is listed once per weather
description, always with identical traffic volume (7,629 extra rows, of which only 17 are exact duplicates). Left
alone, every average, correlation and probability would be weighted towards hours with many weather labels.

All Part 1 statistics therefore use **one row per hour**, kept as the SQL view `traffic_hourly`. The row kept is
the one with the most severe weather label, in this order (most severe first):
Thunderstorm > Squall > Snow > Rain > Drizzle > Fog > Mist > Haze > Smoke > Clouds > Clear.
The Python pipeline in `part2_python` applies the same rule, so SQL and Python give the same numbers.

The holiday label is also filled across the whole calendar day. In the raw data only the 00:00 row of a holiday
carries its name, which makes the flag nearly useless. After filling, there are **1,203 holiday hours across 53
dates**.

## How to run the SQL

The database already contains the raw table `traffic`. The first block of `part1_queries.sql` creates the
`traffic_hourly` view, and every later query reads from that view.

With the SQLite command line:

```bash
cd part1_data_analytics/sql
sqlite3 Task1.db
.read part1_queries.sql
```

or open `Task1.db` in a viewer such as DB Browser for SQLite and paste the queries one at a time.

Checks that should hold after the view is created:

| Check | Expected |
| --- | --- |
| Raw rows in `traffic` | 48,204 |
| Rows in `traffic_hourly` | 40,575 (one per distinct hour) |
| Holiday hours / dates | 1,203 / 53 |
| Hours above 5,500 vehicles | 6,107 (15.1%) |
| Mean traffic per hour | 3,290.65 |

## Main findings

- **Time drives traffic more than weather.** In 2017 the 16:00 peak averages 5,834 vehicles per hour and the
  07:00 peak 4,820, against 386 at 03:00. From 2015 to 2017, Monday to Friday average 3,324 to 3,655 vehicles per
  hour; Saturday 2,816; Sunday 2,458.
- **Temperature is a weak predictor.** The correlation with traffic is 0.14 (r squared about 0.02). Within each hour
  of the day it stays weak.
- **Weather has a modest effect.** Fog (2,654), mist, snow and thunderstorms average below the overall 3,291.
  Congestion (above 5,500 vehicles) is somewhat less likely in clear weather than in clouds (odds ratio 0.74), but the
  link is partly explained by when each weather type occurs.
- **Holidays are quiet.** Whole-day traffic on the holidays analysed is about 25% to 50% below the year's average.
- **Congestion** (above 5,500 vehicles per hour) occurs in about 15% of hours, concentrated in weekday rush hours.

## Data quality issues found

- Nine missing months (September 2014 to May 2015), so yearly totals measure coverage, not demand. Compare
  yearly averages instead.
- 10 temperature readings of 0 K (31 January and 2 February 2014), one rainfall reading of 9,831.3 mm, and two
  zero-traffic hours (18:00 and 23:00 on 23 July 2016). Part 1 reports these; cleaning is done in Part 2.
- A low-traffic stretch around 23 July 2016: that day averages 277 vehicles per hour across 24 readings. Part 2
  showed that the stretch is wider, from 22:00 on 22 July to about 17:00 on 24 July (44 hours). It needs checking
  against road-closure or sensor logs.
- Smoke (2 hours) and Squall (1 hour) are too rare for reliable averages.
- The holiday text "None" means "not a holiday", not a missing value.

## Three congestion definitions, kept distinct

1. **Fixed threshold, policy-facing:** more than 5,500 vehicles per hour (6,107 hours, 15.1%). Used for the
   probability tasks.
2. **Dashboard bands:** Low below 4,500 (26,060 hours), Medium 4,500 to 5,500 (8,408), High above 5,500 (6,107).
3. **Quartiles, a modelling construct:** 1,248 / 3,427 / 4,952 on the unrepaired hourly data, used later in
   Parts 2 and 3. After Part 2's repair of two zero-traffic readings they become 1,249 / 3,428 / 4,952.

## The dashboard

The Power BI page shows three headline numbers (40,575 hourly readings, average traffic 3,291 vehicles, average
temperature 8.2 °C), four charts (daily average 2015 to 2017, average by hour in 2017, average by weather condition,
temperature against traffic) and three slicers (hour range, weather condition, traffic category).

## How this connects to Parts 2 and 3

- `part2_python` reproduces the same one-row-per-hour rule in Python and reconciles with the SQL view (same 40,575
  hours, same holiday and weather labels, same 6,107 congested hours). Its only differences are 13 deliberately
  repaired values.
- `part3_machine_learning` builds its models on the Part 2 output.
