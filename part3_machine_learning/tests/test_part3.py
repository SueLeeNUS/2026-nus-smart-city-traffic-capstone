"""Checks for the rules Part 3 depends on. Run with: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from src import config, data_prep, features, mlops, serving, unsupervised


class CleanedDataRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = data_prep.load_clean()
        cls.frame = features.build_frame(cls.clean)

    def test_one_row_per_hour(self):
        self.assertTrue(self.clean["date_time"].is_unique)
        self.assertEqual(len(self.clean), 40575)

    def test_holiday_fill_across_day(self):
        self.assertEqual(int((self.clean["holiday"] != "No Holiday").sum()), 1203)
        self.assertEqual(self.clean.loc[self.clean["holiday"] != "No Holiday", "date_time"].dt.date.nunique(), 53)

    def test_no_impossible_values_remain(self):
        self.assertGreater(self.clean["temp"].min(), 0)
        self.assertLessEqual(self.clean["rain_1h"].max(), config.MAX_PLAUSIBLE_RAIN_MM)
        self.assertGreater(self.clean["traffic_volume"].min(), 0)

    def test_congestion_rate_matches_part1(self):
        self.assertAlmostEqual((self.clean["traffic_volume"] > 5500).mean(), 0.1505, places=4)

    def test_bands_match_dashboard_definition(self):
        counts = self.frame["congestion_band"].value_counts().to_dict()
        self.assertEqual(counts, {"Low": 26060, "Medium": 8408, "High": 6107})

    def test_split_is_chronological_and_disjoint(self):
        train, test = features.chronological_split(self.frame)
        self.assertTrue((train["date_time"].dt.year <= config.TRAIN_END_YEAR).all())
        self.assertTrue((test["date_time"].dt.year == config.TEST_YEAR).all())
        window = self.frame["in_anomaly_window"] == 1
        self.assertEqual(int(window.sum()), 44)
        self.assertEqual(len(train) + len(test) + int(window.sum()), len(self.frame))  # every row lands in one place

    def test_outage_window_never_reaches_train_or_test(self):
        train, test = features.chronological_split(self.frame)
        for part in (train, test):
            t = part["date_time"]
            self.assertFalse(((t >= "2016-07-22 22:00") & (t <= "2016-07-24 17:00")).any())

    def test_holiday_flag_counts_match_part2(self):
        self.assertEqual(int(self.frame["is_holiday"].sum()), 1203)

    def test_classifier_never_sees_the_label_inputs(self):
        self.assertNotIn("weather_main", features.CLASSIFIER_FEATURES)
        self.assertNotIn("traffic_volume", features.CLASSIFIER_FEATURES)
        self.assertNotIn("snow_1h", features.CLASSIFIER_FEATURES)


class HelperRules(unittest.TestCase):
    def test_psi_is_zero_for_identical_samples(self):
        x = np.random.default_rng(1).normal(size=3000)
        self.assertLess(mlops.psi(x, x), 1e-9)

    def test_psi_flags_a_clear_shift(self):
        rng = np.random.default_rng(2)
        self.assertGreater(mlops.psi(rng.normal(size=3000), rng.normal(1.0, 1, size=1000)), config.PSI_ALERT)

    def test_daypart_boundaries(self):
        self.assertEqual(unsupervised.daypart(3), "overnight")
        self.assertEqual(unsupervised.daypart(7), "morning peak")
        self.assertEqual(unsupervised.daypart(16), "evening peak")
        self.assertEqual(unsupervised.daypart(22), "late evening")

    def test_severity_collapse_keeps_most_severe(self):
        raw = pd.DataFrame({
            "date_time": pd.to_datetime(["2013-01-01 10:00"] * 3),
            "weather_main": ["Clear", "Snow", "Rain"],
            "weather_description": ["sky is clear", "heavy snow", "light rain"],
        })
        raw["_rank"] = raw["weather_main"].map(config.SEVERITY_RANK)
        kept = raw.sort_values("_rank").drop_duplicates("date_time").iloc[0]["weather_main"]
        self.assertEqual(kept, "Snow")


class ServingValidation(unittest.TestCase):
    def test_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            serving.payload_to_frame({"datetime": "2018-03-14 16:00:00", "temp": 280.0})

    def test_rejects_bad_datetime(self):
        with self.assertRaises(ValueError):
            serving.payload_to_frame({"datetime": "14/03/2018", "temp": 280.0,
                                      "weather_main": "Clear", "clouds_all": 40})

    def test_rejects_kelvin_at_or_below_zero(self):
        with self.assertRaises(ValueError):
            serving.payload_to_frame({"datetime": "2018-03-14 16:00:00", "temp": 0.0,
                                      "weather_main": "Clear", "clouds_all": 40})


if __name__ == "__main__":
    unittest.main()
