# Capstone Part 3: Traffic Volume and Congestion-Risk Models

Sue Lee · Smart City Traffic Intelligence capstone · I-94 westbound, Minneapolis–St Paul

This folder builds the Part 3 models on the cleaned hourly file produced in Part 2. It predicts traffic volume,
flags high-risk hours, looks for patterns, monitors drift and runs a small recommendation tool.

## Decisions a reader must know

These rules shape every number in this folder.

1. **The 22–24 July 2016 outage window is excluded from training and testing.** The window runs from 22 July 2016 at
   22:00 to 24 July 2016 at 17:00 (44 hours). In this stretch 28 of 44 hours record fewer than 100 vehicles, against
   7 of about 4,200 July hours outside it. The rows stay in the data file and are flagged `in_anomaly_window = 1`, but
   they never enter the training set or the test set. The run log records the exclusion.
2. **Scalers are fitted on 2017 and earlier only.** Every scaler is fitted on training years (2012–2017, outage
   window excluded) and then applied unchanged to 2018. This applies to the logistic regression and Ridge pipelines,
   the clustering scaler and the LSTM scaler. The random forests do not need scaling.
3. **Thresholds are computed from 2017 and earlier only, with one exception.** The quartile cut-offs are recomputed
   from training years at run time (`features.training_quartiles`): 1,249 / 3,415 / 4,954 vehicles/hour.
   The **high-risk traffic threshold is the brief's fixed value, 3,427 vehicles/hour** (close to, but not exactly,
   the all-years median of 3,428 on the cleaned data). This is the one exception to the rule. The training-years
   median (3,415) differs by 12 vehicles and is not used.
4. **Splits are chronological, never random.** Train on 2012–2017 (outage window excluded), test on 2018.
5. **Holiday label.** Part 2 uses "No Holiday" for ordinary days. The feature `is_holiday` treats both "No Holiday"
   and the older text "None" as not a holiday, so a holiday cannot be missed by a label mismatch.

## The high-risk label

`high_risk` equals 1 when **both** conditions hold in the same hour (confirmed by the instructor):

- **Traffic:** volume above 3,427 vehicles/hour (the top half of the hourly data, not the top quartile).
- **Weather:** Thunderstorm, Snow or Squall, or low visibility (Fog, Mist, Haze or Smoke).

Each hour keeps its most severe weather label (the Part 2 rule). This gives **3,133 of 40,575 hours (7.7%)**, as
the instructor confirmed. A first-row-in-file rule would give 2,964 (7.3%), which is the figure in the brief's example;
it is not used here.

## Inputs and outputs

- **Input:** `data/processed/traffic_clean.csv`, copied unchanged from the Part 2 output (40,575 hourly rows, 9 columns).
- **Classifier features:** time (hour, day of week, month, weekend, holiday) and temperature, rainfall and cloud cover.
  The weather category, `traffic_volume` and `snow_1h` are excluded because they define the label.
- **Regressor features:** the same time features plus temperature, rainfall, snow, cloud cover and the weather category.
- **Feature flags:** `is_precipitation` (the weather label says precipitation) and `is_raining` (rain was measured)
  are different. 6,627 hours carry a precipitation label but record 0 mm of rain. Part 3 uses the measured `rain_1h`
  directly.

## Results on the 2018 test year

| Task | Model | Result |
|---|---|---|
| Traffic volume (vehicles/h) | Random forest | MAE 224.0, R² 0.962 |
| | Naive hour-of-week baseline | MAE 264.5, R² 0.944 |
| | Ridge | MAE 829.6, R² 0.716 |
| High risk (594 positives, 9.1%) | Random forest | PR-AUC 0.413, precision 0.320, recall 0.727, ROC-AUC 0.884 |
| | Logistic regression | PR-AUC 0.284 |
| | Time-only random forest | PR-AUC 0.262 |
| | Never predict high risk | Accuracy 90.9% (shown only as the baseline) |
| Traffic volume, deep learning | LSTM (24-hour window) | MAE 229.2, vs. a persistence baseline (last hour repeated) of 588.4 |

Accuracy is not used to judge the classifier, because the classes are unbalanced. PR-AUC, precision and recall are the
main measures (figure: `outputs/figures/classifier_precision_recall.png`).

The pipeline was also run on Google Colab, with PyTorch, SHAP and MLflow installed, to check every part the local
environment here could not run. The regression and classification figures matched closely (random forest MAE 224.2
vs. 224.0 here, classifier PR-AUC 0.409 vs. 0.413 here); the small gap is normal cross-machine variation in a Random
Forest and does not change any conclusion. The high-risk count (594), the clustering (k = 7, same cluster traffic
means), and the association rules (same top three, same percentages) matched exactly.

## Deep learning and explainability (run on Colab)

An LSTM (24-hour input window, one hidden layer of 32 units, 5 epochs) predicts traffic volume from the same
training-years-only scaling as the other models (`StandardScaler` fitted on 2012–2017, outage window excluded).

- **Accuracy:** MAE 229.2 vehicles/hour on 2018, against 588.4 for a naive "same as the last hour" guess. This is
  close to the Random Forest's 224.0, not better. A simple tabular model matched a deep-learning model trained
  end to end, which is worth noting in the Responsible and Sustainable AI section: the heavier model did not buy
  more accuracy here, only more training time and complexity.
- **Explanation:** SHAP cannot look inside an LSTM directly, so a small Gradient Boosting surrogate is trained to
  imitate the LSTM's predictions (fidelity R² = 0.995 against the LSTM's own output on 2018), and SHAP explains the
  surrogate instead.
- **What drives the prediction**, ranked by mean absolute SHAP importance:

  | Feature | Importance |
  |---|---|
  | Hour of day (cosine) | 0.428 |
  | Traffic volume, previous hour | 0.423 |
  | Hour of day (sine) | 0.133 |
  | Weekday/weekend, previous hour | 0.053 |
  | 24-hour mean traffic | 0.018 |
  | 24-hour traffic spread (std) | 0.009 |
  | Temperature, previous hour | 0.008 |
  | 24-hour max traffic | 0.005 |
  | Cloud cover, previous hour | 0.005 |
  | 24-hour min traffic | 0.002 |
  | Rainfall, previous hour | 0.0004 |

  Time of day and the previous hour's traffic together account for most of the model's decisions. Every weather
  input sits at the bottom, with rainfall essentially unused. This matches the Part 1 finding that traffic follows
  the clock and calendar far more than the weather, now confirmed inside a deep-learning model given every input.

Outputs: `outputs/metrics/lstm_metrics.csv`, `outputs/metrics/lstm_surrogate_importance.csv`.

## Other parts

- **Congestion clusters:** k-means with k = 7, chosen by silhouette score on training-year scaling.
- **Association rules:** 20 rules predicting high congestion (above 5,500 vehicles/h). The strongest links evening and
  morning peaks on weekdays to high congestion, at 3.7–4.1 times the base rate of 15%.
- **Responsible AI:** coverage by year, and error by weekday/weekend, daytime/night and weather group
  (`outputs/metrics/bias_report_2018.csv`).
- **Drift monitor (2018, by month):** compares each month with the same calendar month in the training years.
  Traffic volume stays stable. Cloud cover is reported as a coding check, because its share of code "1" rises
  in every 2018 month. A month with no rain recorded is flagged as a data gap (January to July 2018).
  Temperature and rain shifts are reported as shifts, not coding faults. With cloud cover excluded from the drift
  decision, January, March, April, May and June 2018 still raise an ALERT; this is driven mainly by the model's
  error rising against the 2017 reference (for example +29.6% in January), not by the cleaned data itself drifting.
- **Experiment tracking:** every supervised, clustering and drift run is logged to MLflow (`mlruns_local/`), 7 runs
  in total across three experiments (`traffic-supervised`, `traffic-unsupervised`, `traffic-mlops`), confirmed
  `FINISHED` on a full Colab run. If MLflow is not installed, or fails to start for any reason, the project falls
  back to a local JSON-lines log (`mlruns_local/runs.jsonl`) automatically, so the pipeline never stops because of
  the tracking layer.
- **Recommendation tool:** `python -m src.cli recommend ...` (inputs are validated; the note in the output says the
  high-risk probability uses temperature, rainfall, cloud cover and time only).

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib joblib   # core dependencies
python run_all.py                  # full pipeline; logs to logs/pipeline.log
python -m unittest discover -s tests -v
```

Deep learning and MLflow could not run in the environment this project was written in (no PyPI access there), so
they were run separately on Google Colab (`!pip install shap mlflow`; PyTorch is pre-installed there) and the
results above are from that run. The notebook used for that run is included at `colab/Part3_DeepLearning_MLflow.ipynb`
— it installs the extra packages, uploads this project, runs the tests and the full pipeline, and prints the
deep-learning and MLflow results. To repeat it: open that notebook in Colab, run it, and upload this folder
(as a zip) when prompted. The two saved `.skops` model files shared by a course mate are a different project's
output and are not used here.

**A version note for anyone re-running this on Colab or a newer machine:** MLflow 2.x+ refuses its plain-folder
tracking backend by default and raises an error instead of simply not being available. `src/tracking.py` sets
`MLFLOW_ALLOW_FILE_STORE=true` itself and catches any MLflow start-up failure, not only a missing installation, so
this should not need any manual fix. If an MLflow error still stops the whole pipeline, check that `src/tracking.py`
has this fix (search the file for `MLFLOW_ALLOW_FILE_STORE`).

## Limitations

- About nine months are missing (September 2014 to May 2015) and 2012, 2014 and 2015 are under 60% covered.
- Part 2 keeps the most severe weather label per hour, which favours bad-weather hours.
- The classifier raises some alerts on clear hours, because it cannot see the weather category.
- The 2018 rain record is empty from January to July, so rain results cover August and September only.
