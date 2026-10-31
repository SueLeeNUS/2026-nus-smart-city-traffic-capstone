---
title: "Part 3 Report: Traffic Volume and Congestion-Risk Models"
---

# Part 3 Report: Traffic Volume and Congestion-Risk Models

Sue Lee · Smart City Traffic Intelligence capstone · I-94 westbound, Minneapolis–St Paul

## 1. Purpose

Part 1 found that traffic follows the clock and calendar far more than the weather. Part 2 turned that into a
clean, feature-rich hourly file. Part 3 builds on it directly: it predicts traffic volume, flags hours at high
congestion risk under adverse weather, looks for patterns a human would not spot by eye, and adds the monitoring
a model needs once it is in use. Every decision that could leak information from the future into training is
documented and tested.

## 2. Data contract with Part 2

Part 3 reads `traffic_clean.csv` from Part 2 unchanged — no re-cleaning happens here. Three rules apply on top of
it, and they are checked by a 16-test suite before any model runs:

- **The 22–24 July 2016 outage window** (44 hours, 22 July 22:00 to 24 July 17:00) is excluded from both training
  and test. Part 2 flagged this stretch as likely sensor or road failure; the two repaired zero-traffic readings
  inside it are placeholders, not measurements, so no model is trained or scored on them.
- **The train/test split is chronological, never random**: train on 2012–2017, test on 2018. Hourly traffic is
  autocorrelated, so a random split would leak adjacent hours between the two sets and inflate every score.
- **Every scaler and threshold is fitted on training years only** (2012–2017, outage window excluded), including
  the classifier's and regressor's scalers, the clustering scaler, and the quartile cut-offs used for the
  congestion category. The one documented exception is the high-risk traffic threshold, fixed at the instructor's
  confirmed value of 3,427 vehicles/hour (see Section 4).

## 3. Supervised learning

**Traffic volume (regression).** Three models were compared on the 2018 test year:

| Model | MAE (vehicles/h) | R² |
|---|---|---|
| Naive hour-of-week baseline | 264.5 | 0.944 |
| Ridge regression | 829.6 | 0.716 |
| Random forest | **224.0** | **0.962** |

The random forest is the best model and beats the naive baseline, which already captures most of the daily
pattern. Ridge performs worse than the naive baseline, because a straight line cannot capture how sharply volume
changes between rush hour and overnight.

**High-risk hours (classification).** An hour is labelled `high_risk` when both traffic is above 3,427
vehicles/hour (the top half of the data) and the weather is adverse (Thunderstorm, Snow or Squall, or low
visibility — Fog, Mist, Haze or Smoke). This is a deliberately rare event: 594 of 6,533 test hours (9.1%), so a
model that never predicts high risk would already score 90.9% accuracy. Accuracy is therefore not used to judge
the model; precision, recall and PR-AUC are.

| Model | Precision | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logistic regression | 0.223 | 0.678 | 0.284 | 0.811 |
| Random forest | **0.320** | **0.727** | **0.413** | **0.884** |
| Random forest, time features only (ablation) | 0.239 | 0.717 | 0.262 | 0.832 |

The random forest with the full feature set clearly outperforms the time-only ablation (PR-AUC 0.413 vs. 0.262),
showing that temperature, rainfall and cloud cover add real information beyond the hour and day of week alone.
The classifier never sees the weather category, `traffic_volume` or `snow_1h` directly, because these define the
label; it is given only time features plus the measured temperature, rainfall and cloud cover.

## 4. The high-risk label

`high_risk` was confirmed with the instructor as: traffic above the median (3,427 vehicles/hour, the top half of
the data) **and** adverse weather (Thunderstorm, Snow, Squall, Fog, Mist, Haze or Smoke) in the same hour. Using
Part 2's rule of keeping the most severe weather label per hour, this gives **3,133 of 40,575 hours (7.7%)**.

## 5. Unsupervised learning

**Clustering.** K-means on cyclical time features, weather severity and traffic volume, with k chosen by
silhouette score, selects **k = 7**. Each cluster was named from its profile:

| Cluster | Share of hours | Mean volume (veh/h) |
|---|---|---|
| Overnight, weekday, quiet, clear-to-cloudy | 18.3% | 851 |
| Overnight, weekday, quiet, adverse weather | 11.0% | 979 |
| Late evening, weekday, moderate, clear-to-cloudy | 16.7% | 2,569 |
| Evening peak, weekday, busy, adverse weather | 7.4% | 4,130 |
| Morning peak, weekday, busy, clear-to-cloudy | 18.1% | 4,766 |
| Morning peak, weekday, busy, adverse weather | 9.2% | 4,805 |
| Evening peak, weekday, peak, clear-to-cloudy | 19.2% | 5,132 |

The clusters separate cleanly by time of day and volume first, with weather as a secondary split within each
time band — consistent with Part 1's finding that time dominates.

**Association rules.** Mining itemsets of time, day type and weather against the fixed congestion threshold (above
5,500 vehicles/hour, base rate 15.1%) finds 20 rules. The strongest:

- Evening peak on a weekday: high congestion in **62%** of hours, **4.12×** the base rate.
- Morning peak on a weekday: high congestion in **56%** of hours, **3.72×** the base rate.
- Morning peak with low visibility: high congestion in **47%** of hours, **3.14×** the base rate.

Weather only strengthens a rule that time already sets up; no weather-only rule reaches a comparable lift.

## 6. Deep learning and explainability

An LSTM (24-hour input window, one hidden layer of 32 units, 5 training epochs) was trained on the same
training-years-only scaling as the other models. On the 2018 test year it scores **MAE 229.2 vehicles/hour**,
against 588.4 for a naive "same as the last hour" guess — a real improvement, but essentially tied with the
random forest's 224.0, not better, despite far more training cost.

SHAP cannot explain an LSTM directly, so a Gradient Boosting surrogate was trained to imitate its predictions
(fidelity R² = 0.995 against the LSTM's own output), and SHAP explains the surrogate instead. By mean absolute
importance, the hour of day and the previous hour's traffic volume together account for the large majority of the
model's decisions (0.43 + 0.42 + 0.13 combined); every weather input sits near zero, with rainfall at 0.0004. This
is the same finding as Part 1, now confirmed inside a model given every available input: traffic follows the
clock, not the weather.

## 7. MLOps

**Experiment tracking.** Every supervised, clustering and drift run is logged to MLflow: 7 runs across three
experiments (`traffic-supervised`, `traffic-unsupervised`, `traffic-mlops`), all `FINISHED`. If MLflow is
unavailable, the project falls back to a local JSON-lines log automatically, so tracking never stops the pipeline.

**Model versioning.** A model version table records each production model, its test-year score and its training
data cut-off, so any served prediction can be traced back to the model that made it.

**Drift monitoring.** Each 2018 month is compared with the same calendar month in the training years, using the
Population Stability Index (PSI) and the change in error against a 2017 reference (MAE 239.8–240.2, depending on
the run). Cloud cover is checked separately as a possible recording-format change, since its code "1" share rises
in every 2018 month, and a month with no rain recorded at all is flagged as a data gap rather than drift — true of
January to July 2018. With those two excluded from the decision, the traffic-volume distribution itself stays
stable all year (PSI below 0.04 every month); most of the remaining ALERTs come from the model's own error rising
against the reference, not from the cleaned data drifting.

**Recommendation and serving.** A command-line tool recommends low-traffic hours for a given day, built on the
same models, and a small FastAPI wrapper validates a request (missing fields, an unparsable date, an impossible
temperature) before predicting. Both state plainly that the high-risk probability is based only on time,
temperature, rainfall and cloud cover, since the classifier never sees the weather category directly.

## 8. Limitations

- About nine months of the underlying record are missing (September 2014 to May 2015), and 2012, 2014 and 2015
  are each under 60% covered, so the model leans more heavily on 2016–2017 than on the full history.
- The classifier raises some false alarms on clear or cloudy hours, because it is deliberately not given the
  weather category, only measured temperature, rainfall and cloud cover.
- 2018's rain record is empty from January to July, so any rain-related comparison for that year only reflects
  August and September.
- The deep-learning and MLOps-tracking stages could not run on the development machine (no package access there)
  and were verified separately on Google Colab; the results above are from that run.

## 9. Conclusion

A simple random forest matches or beats a deep-learning model on this data, at a fraction of the training cost —
the clearest, most reusable finding of this part. Time and recent traffic drive almost every prediction; weather
adds a smaller, genuine signal on top, visible only once time is accounted for. The pipeline is reproducible
(chronological split, training-years-only scaling, a confirmed label definition) and monitored (drift, version
tracking), so its limitations are documented rather than hidden.
