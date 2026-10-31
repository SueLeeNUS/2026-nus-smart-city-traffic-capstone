# Smart City Traffic Intelligence: I-94 Westbound Capstone

Sook Jing Lee · NUS AI, ML and Data Science Programme (2026)

A three-part analysis of hourly traffic volume and weather on the I-94 westbound interstate between Minneapolis
and St Paul, October 2012 to September 2018 (48,204 raw rows, **40,575 distinct hours**). Part 1 explores the data
with SQL, statistics and a dashboard. Part 2 turns the findings into a reproducible Python pipeline. Part 3 builds
and checks predictive models on the cleaned data.

## Repository structure

| Folder | What it holds | Start with |
| --- | --- | --- |
| [`part1_data_analytics/`](part1_data_analytics/) | SQL queries and database, statistical and probability analysis, Power BI dashboard, insights report | its [README](part1_data_analytics/README.md) |
| [`part2_python/`](part2_python/) | Cleaning pipeline, feature engineering, five figures, a command-line app, logging | its [README](part2_python/README.md) |
| [`part3_machine_learning/`](part3_machine_learning/) | Traffic-volume and congestion-risk models, clustering, association rules, deep learning, drift monitor, bias report | its [README](part3_machine_learning/README.md) |

Each part has its own README with the full detail. This page gives the decisions that connect them and the headline
results.

## Decisions that apply to the whole project

1. **One row per hour.** The raw file lists the same hour once per weather description, always with identical
   traffic volume. Every part uses one row per hour, keeping the row with the most severe weather label
   (Thunderstorm > Squall > Snow > Rain > Drizzle > Fog > Mist > Haze > Smoke > Clouds > Clear). Part 1 does this in
   SQL (the view `traffic_hourly`) and Part 2 does it in Python. The two agree on the hours, the weather and holiday
   labels and the 6,107 congested hours; the only differences are the 13 values Part 2 deliberately repairs.
2. **Holiday flag filled across the whole day.** The raw flag marks only 00:00 of a holiday. Filling it gives 1,203
   holiday hours across 53 dates.
3. **Impossible values repaired in Part 2, with a log.** 10 temperatures of 0 K and one 9,831 mm rainfall reading are
   replaced with the median of the same calendar month (pooled across years). Two zero-traffic hours use the
   same-month-and-hour median.
4. **A suspect stretch is excluded from modelling.** From 22:00 on 22 July 2016 to about 17:00 on 24 July 2016
   (44 hours), 28 hours record fewer than 100 vehicles, against 7 of about 4,200 July hours elsewhere. It is probably a
   sensor or road outage. Part 3 keeps the rows but excludes them from training and testing.
5. **Chronological splits and no leakage.** Part 3 trains on 2012 to 2017 and tests on 2018. Scalers and thresholds
   are fitted on the training years only. A random split would let the model see hours next to the ones it is tested
   on.
6. **Three congestion definitions, kept distinct.** (a) Fixed threshold of more than 5,500 vehicles an hour, used for
   policy-facing probabilities (15.1% of hours). (b) Dashboard bands: Low below 4,500, Medium 4,500 to 5,500, High above
   5,500. (c) Quartiles of volume, a modelling construct: 1,248 / 3,427 / 4,952 on the unrepaired data in Part 1 and
   1,249 / 3,428 / 4,952 after Part 2's repair of two zero readings.

## The proxy-label decision (Part 3)

The dataset has no accident or incident column, so **accident risk cannot be observed**. Part 3 predicts a proxy
label instead:

> `high_risk = 1` when, in the same hour, traffic volume is above **3,427 vehicles per hour** (the brief's fixed
> value) **and** the weather is Thunderstorm, Snow or Squall, or low visibility (Fog, Mist, Haze or Smoke).

What this means for anyone reading the results:

- The label is **built from traffic and weather**, not recorded from real crashes. It measures "congested hour in
  poor conditions", not accidents.
- A model that predicts it from time and weather partly re-discovers the rule that created it. For that reason the
  classifier does not receive the weather category, `traffic_volume` or snow, which define the label.
- It marks **3,133 of 40,575 hours (7.7%)**, and **594 of 6,533 test hours (9.1%)** in 2018. The brief's example
  quotes 7.3% (2,964 hours) using the first row in the file for each hour. This project keeps the most severe weather
  per hour, as in Part 2, and the instructor confirmed the 3,133 count.
- Because only about one hour in eleven is positive, accuracy is misleading (always answering "no" scores 90.9% on
  the test year). The classifier is judged on precision, recall and PR-AUC.

## Headline results

| Part | Result |
| --- | --- |
| 1 | Traffic follows the clock and calendar more than the weather. The 16:00 peak in 2017 averages 5,834 vehicles an hour against 386 at 03:00. Temperature correlates only 0.14 with traffic. About 15% of hours exceed 5,500 vehicles. |
| 2 | 48,204 raw rows become 40,575 clean hourly rows and 43 feature columns. Five figures, a five-command analytics app and a logged pipeline. |
| 3, volume | Random forest on 2018: MAE 224.0 vehicles an hour, R² 0.962, against a naive hour-of-week baseline of 264.5. An LSTM reaches MAE 229.2 and does not beat the random forest. |
| 3, risk | Random forest: PR-AUC 0.413, precision 0.320, recall 0.727 on 594 positives. Time-only random forest scores 0.262, logistic regression 0.284. |
| 3, explanation | SHAP on the LSTM surrogate ranks hour of day and the previous hour's traffic far above any weather input, which agrees with Part 1. |

## How to run

```bash
# Part 1: open sql/Task1.db and run sql/part1_queries.sql (see its README)

# Part 2
cd part2_python
pip install -r requirements.txt
python main.py                 # clean -> features -> figures (logs to pipeline.log)
python cli_app/traffic_app.py compare

# Part 3
cd part3_machine_learning
pip install pandas numpy scikit-learn matplotlib joblib
python run_all.py
python -m unittest discover -s tests -v
```

The deep-learning, SHAP and MLflow steps in Part 3 were run on Google Colab; see the Part 3 README for the details.

## Limitations

- About nine months are missing (September 2014 to May 2015), so yearly totals measure coverage, not demand.
- Keeping one weather label per hour favours bad-weather hours and discards the less severe label.
- The 2018 rain record is empty from January to July, so rain results for 2018 cover August and September only.
- The risk label is a proxy and says nothing about real accidents. Smoke (2 hours) and squall (1 hour) are too rare
  for reliable averages.
- The 22 to 24 July 2016 stretch still needs checking against road-closure or sensor logs from the data owner.
