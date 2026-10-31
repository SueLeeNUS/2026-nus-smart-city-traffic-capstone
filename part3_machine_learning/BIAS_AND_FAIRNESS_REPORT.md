---
title: "Part 3: Bias and Fairness Report"
---

# Bias and Fairness Report

Sue Lee · Smart City Traffic Intelligence capstone · I-94 westbound, Minneapolis–St Paul

All numbers below come from the Part 3 pipeline run on the Part 2 cleaned file. The test year is 2018; the
22–24 July 2016 outage window (44 hours) is excluded from training and test throughout.

## 1. Where the data is thin

| Year | Hours recorded | Coverage of a full year |
|---|---|---|
| 2012 | 2,103 | 24% |
| 2013 | 7,294 | 83% |
| 2014 | 4,501 | 51% |
| 2015 | 3,593 | 41% |
| 2016 | 7,838 | 90% |
| 2017 | 8,713 | 99.5% |
| 2018 | 6,533 | 75% (Jan–Sep only) |

2012, 2014 and 2015 are each under 60% covered, and nine months (September 2014 to May 2015) are missing
entirely. The model is therefore trained mostly on 2016–2017 behaviour; it has very little evidence about any
systematic difference in earlier years' traffic patterns. A single location (one stretch of interstate in
Minnesota) also means the model has no evidence at all about how its patterns would transfer elsewhere.

## 2. The label is a judgement, not a fact

`high_risk` is defined as traffic above 3,427 vehicles/hour **and** adverse weather (Thunderstorm, Snow, Squall,
or low visibility — Fog, Mist, Haze, Smoke), confirmed with the instructor. It encodes two choices that are not
neutral:

- **Which weather counts as adverse.** Drizzle and ordinary rain are not included, only the more severe or
  visibility-reducing categories. A different, equally reasonable choice of categories would change which hours
  are labelled high risk and shift every downstream metric.
- **Which hour's weather is kept.** Part 2 keeps the most severe weather label when an hour has several readings.
  This favours bad-weather hours in every calculation that uses the weather category, including this label.

The classifier is also deliberately blind to the weather category itself — it sees only time, temperature,
rainfall and cloud cover, not `weather_main` — because the category defines the label. This is a correct design
choice to avoid the model simply learning the label's own definition, but it is also the direct cause of the
false alarms described below.

## 3. Uneven performance across groups (2018 test year)

| Group | Hours | Recall | False-positive rate | Volume MAE (veh/h) | Mean error (veh/h) |
|---|---|---|---|---|---|
| Weekday | 4,671 | 0.76 | 0.19 | 206 | +11 |
| Weekend | 1,862 | 0.55 | 0.08 | 269 | −0.4 |
| Daytime 06–18 | 3,544 | 0.75 | 0.31 | 272 | +13 |
| Night 19–05 | 2,989 | 0.00 | 0.00 | 168 | +2 |
| Clear or cloudy | 4,324 | n/a (no positives) | 0.15 | 206 | −30 |
| Rain, drizzle or storm | 1,089 | 0.58 | 0.25 | 190 | +45 |

**Weekends are under-served.** Recall is 0.55 at weekends against 0.76 on weekdays — weekend high-risk hours are
more often missed than caught.

**The classifier raises false alarms on clear hours.** About 15% of clear or cloudy hours are flagged high risk,
because the model cannot see the weather category and must infer risk from temperature, rain and cloud cover
alone. For a traffic advisory system, a false alarm on a clear day is a direct trust cost.

**Night hours are effectively untouched.** Only 0.5% of night hours are labelled high risk, and the classifier
catches none of them — this label is, in practice, a daytime tool.

**Regression error does not spike in bad weather.** The volume regressor's error in rain, drizzle or storm
conditions (MAE 190) is lower than on an ordinary weekday (MAE 206). Its bias runs in opposite directions by
condition: it over-predicts in rain (+45 vehicles/hour) and under-predicts in clear or cloudy conditions
(−30 vehicles/hour). Neither error is large, but a system that systematically over-predicts congestion in rain
could trigger unnecessary advisories, and under-predicting on clear days could miss a genuine build-up.

## 4. Stability over time

Each 2018 month was compared with the same calendar month in training years, using the Population Stability
Index (PSI) and the change in error against a 2017 reference (MAE 239.8–240.2 vehicles/hour, depending on the
run). Two confounds were separated from genuine drift before judging stability:

- **Cloud cover recording looks to have changed.** Its PSI reaches 1.0–1.4 in several months, driven by a rising
  share of the code "1" in every 2018 month compared with the same month in training years. This reads as a
  change in how the value was recorded, not a change in the weather, so it is excluded from the drift decision
  and reported separately as a coding check.
- **2018's rain record is empty from January to July.** No rainfall at all was recorded in these months, against
  2–12% of hours with rain in the same months of earlier years. This is a data gap, not evidence that it did not
  rain; it is flagged separately, and any rain-related figure for 2018 should be read as covering August and
  September only.

With those two excluded, **traffic volume itself stays stable all year** (PSI under 0.04 every month). Most of
the remaining ALERT months are driven by the model's own prediction error moving against the 2017 reference
(for example +29.7% in January, +31.9% in April), not by the underlying traffic pattern changing. This points to
a model-stability issue worth monitoring, more than a change in real-world traffic.

## 5. Governance recommendations

- **Treat any output as advisory, not a trigger for automated action**, until the weekend recall gap and the
  false-alarm rate on clear hours are addressed — neither is currently acceptable for an automated system that
  controls signs, tolls or lane direction.
- **Do not use this model, unmodified, outside the I-94 corridor or outside a Minnesota-like climate.** The
  training data gives no evidence either way about transfer to another location.
- **Retrain or re-examine when the drift monitor raises an ALERT on traffic volume itself** (not on cloud cover
  or a rain gap), or when weekend recall falls further.
- **Keep a human in the loop for any public-facing alert.** A transport operations centre operator should be
  able to review and override a flagged hour before it reaches the public, given the known false-alarm rate.

## 6. What this report does not cover

- Energy or compute cost of training (not measured; see the main Part 3 report for the LSTM-versus-Random-Forest
  comparison, which found no accuracy benefit from the heavier model).
- Error broken down by the onset and end of rush hour specifically (not measured separately from the broader
  daytime/night split above).
- Any comparison with a model trained on a different city or climate (no such data available).
