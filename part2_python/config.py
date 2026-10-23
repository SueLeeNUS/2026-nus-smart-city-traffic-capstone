"""Project-wide constants: the one place where shared decisions live.

Part 3 imports (or copies) these values so that Parts 1, 2 and 3 agree.
"""

# One constant random seed, set everywhere randomness is used (Part 3 splits, k-means, model fits).
RANDOM_SEED = 42

# GRAIN RULE: the raw file lists the same hour once per weather description, always with
# identical traffic volume. We keep ONE row per hour: the row whose weather_main is the most
# severe, using this order (most severe first). The SQL view traffic_hourly in Part 1 uses
# the same order, so SQL and Python give the same numbers.
WEATHER_SEVERITY_ORDER = [
    "Thunderstorm", "Squall", "Snow", "Rain", "Drizzle",
    "Fog", "Mist", "Haze", "Smoke", "Clouds", "Clear",
]

# Weather groups used for the proxy accident-risk label in Part 3.
SEVERE_WEATHER = ["Thunderstorm", "Snow", "Squall"]
LOW_VISIBILITY_WEATHER = ["Fog", "Mist", "Haze", "Smoke"]
PRECIPITATION_WEATHER = ["Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"]

# Three congestion definitions exist in this project. They are kept separate on purpose:
#   1. Fixed threshold, policy-facing (Part 1, probability tasks): volume above 5,500.
#   2. Bands, dashboard-facing (Part 1, Power BI): Low below 4,500, Medium 4,500-5,500, High above.
#   3. Quartiles, modelling construct (Parts 2 and 3): cut-offs computed from the data.
CONGESTION_FIXED_THRESHOLD = 5500
