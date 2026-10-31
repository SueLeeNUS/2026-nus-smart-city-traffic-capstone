-- ===== SETUP: one row per hour (fixes duplicate timestamps) =====
-- Same hour appears once per weather description, with identical traffic volume.
-- Rule: keep ONE row per hour, the most severe weather_main
-- (Thunderstorm > Squall > Snow > Rain > Drizzle > Fog > Mist > Haze > Smoke > Clouds > Clear).
-- Holiday label is also filled across the whole calendar day (it was only on 00:00).
DROP VIEW IF EXISTS traffic_hourly;
CREATE VIEW traffic_hourly AS
WITH ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY date_time
           ORDER BY CASE weather_main
                      WHEN 'Thunderstorm' THEN 1 WHEN 'Squall'  THEN 2
                      WHEN 'Snow'         THEN 3 WHEN 'Rain'    THEN 4
                      WHEN 'Drizzle'      THEN 5 WHEN 'Fog'     THEN 6
                      WHEN 'Mist'         THEN 7 WHEN 'Haze'    THEN 8
                      WHEN 'Smoke'        THEN 9 WHEN 'Clouds'  THEN 10
                      ELSE 11 END,
                    weather_description) AS rn
  FROM traffic
),
hol AS (
  SELECT DISTINCT date(date_time) AS d, holiday AS hol_name
  FROM traffic WHERE holiday <> 'None'
)
SELECT r.date_time, r.temp, r.rain_1h, r.snow_1h, r.clouds_all,
       r.weather_main, r.weather_description, r.traffic_volume,
       COALESCE(h.hol_name, 'None') AS holiday
FROM ranked r
LEFT JOIN hol h ON date(r.date_time) = h.d
WHERE r.rn = 1;

-- Task 1.1: verify the data loaded correctly (raw table)
-- (a) total row count: expect 48204
SELECT COUNT(*) AS total_rows FROM traffic;

-- (b) preview the first 5 rows
SELECT * FROM traffic LIMIT 5;

-- (c) date range: expect 2012-10-02 to 2018-09-30
SELECT MIN(date_time), MAX(date_time) FROM traffic;

-- (d) data types: expect real, integer
SELECT typeof(temp), typeof(traffic_volume) FROM traffic LIMIT 1;

-- (e) duplicate timestamps in raw data: expect 7629
SELECT COUNT(*) - COUNT(DISTINCT date_time) AS dup_timestamps FROM traffic;

-- (f) hourly view: expect 40575 rows and 40575 distinct hours (one row per hour)
SELECT COUNT(*) AS hourly_rows, COUNT(DISTINCT date_time) AS distinct_hours FROM traffic_hourly;

-- (g) holiday fill check: expect 1203 holiday hours across 53 dates
SELECT COUNT(*) AS holiday_hours, COUNT(DISTINCT date(date_time)) AS holiday_dates
FROM traffic_hourly WHERE holiday <> 'None';

-- Task 1.2: yearly totals and change (2012-2017), from hourly data
-- NOTE: raw yearly SUM depends on how many hours were recorded, so avg_per_hour is shown beside it.
WITH yearly AS (
  SELECT strftime('%Y', date_time) AS yr,
         COUNT(*)            AS hours_recorded,
         SUM(traffic_volume) AS total_volume,
         ROUND(AVG(traffic_volume)) AS avg_per_hour
  FROM traffic_hourly
  WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
  GROUP BY yr
)
SELECT yr, hours_recorded, total_volume, avg_per_hour,
       total_volume - LAG(total_volume) OVER (ORDER BY yr) AS change_total,
       ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY yr))
             / LAG(total_volume) OVER (ORDER BY yr), 1)     AS pct_change
FROM yearly ORDER BY yr;

-- Task 1.3: whole-day average temperature and traffic on New Years Day and Labor Day
-- 1 Jan 2015 does not exist (data missing Sep 2014 - May 2015), so 2014 is used as the
-- substitute year, with 2016, 2017 and 2018 shown for context.
SELECT strftime('%Y', date_time) AS yr, holiday,
       COUNT(*)                      AS hours,
       ROUND(AVG(temp), 2)           AS avg_temp_K,
       ROUND(AVG(temp) - 273.15, 1)  AS avg_temp_C,
       ROUND(AVG(traffic_volume), 0) AS avg_traffic
FROM traffic_hourly
WHERE holiday IN ('New Years Day', 'Labor Day')
  AND strftime('%Y', date_time) IN ('2014','2015','2016','2017','2018')
GROUP BY yr, holiday
ORDER BY holiday, yr;

-- Task 2.1a: mean, min, max, range and sample variance of traffic_volume
SELECT COUNT(*)                                   AS n,
       ROUND(AVG(traffic_volume), 2)              AS mean_volume,
       MIN(traffic_volume)                        AS min_volume,
       MAX(traffic_volume)                        AS max_volume,
       MAX(traffic_volume) - MIN(traffic_volume)  AS range_volume,
       ROUND((AVG(traffic_volume * traffic_volume)
              - AVG(traffic_volume) * AVG(traffic_volume))
             * COUNT(*) / (COUNT(*) - 1), 2)      AS variance_sample
FROM traffic_hourly;

-- Task 2.1b: median of traffic_volume
SELECT AVG(traffic_volume) AS median_volume
FROM (SELECT traffic_volume FROM traffic_hourly
      ORDER BY traffic_volume
      LIMIT 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM traffic_hourly));

-- Task 2.1c: standard deviation = square root of the sample variance
SELECT ROUND(SQRT((AVG(traffic_volume * traffic_volume)
                   - AVG(traffic_volume) * AVG(traffic_volume))
                  * COUNT(*) / (COUNT(*) - 1)), 2) AS std_dev
FROM traffic_hourly;

-- Task 2.2a: Pearson correlation between temp and traffic_volume (all hours)
SELECT ROUND(
  (AVG(temp * traffic_volume) - AVG(temp) * AVG(traffic_volume)) /
  SQRT((AVG(temp * temp) - AVG(temp) * AVG(temp)) *
       (AVG(traffic_volume * traffic_volume) - AVG(traffic_volume) * AVG(traffic_volume)))
, 4) AS correlation_r
FROM traffic_hourly;

-- Task 2.2b: correlation excluding 0 K (impossible) temperature readings
SELECT ROUND(
  (AVG(temp * traffic_volume) - AVG(temp) * AVG(traffic_volume)) /
  SQRT((AVG(temp * temp) - AVG(temp) * AVG(temp)) *
       (AVG(traffic_volume * traffic_volume) - AVG(traffic_volume) * AVG(traffic_volume)))
, 4) AS correlation_r_excl_zero
FROM traffic_hourly
WHERE temp > 0;

-- Task 2.2c: correlation between temp and traffic within each hour of the day
-- (holds time of day fixed, to check the daily-cycle confound)
SELECT CAST(strftime('%H', date_time) AS INTEGER) AS hr,
       ROUND((AVG(temp * traffic_volume) - AVG(temp) * AVG(traffic_volume)) /
             SQRT((AVG(temp * temp) - AVG(temp) * AVG(temp)) *
                  (AVG(traffic_volume * traffic_volume) - AVG(traffic_volume) * AVG(traffic_volume)))
       , 3) AS r
FROM traffic_hourly
WHERE temp > 0
GROUP BY hr
ORDER BY hr;

-- Task 3.1: P(Congestion), P(Clear), P(Congestion AND Clear)
SELECT COUNT(*)                                             AS total_hours,
       SUM(traffic_volume > 5500)                           AS congestion_hours,
       SUM(weather_main = 'Clear')                          AS clear_hours,
       SUM(traffic_volume > 5500 AND weather_main = 'Clear') AS both_hours,
       ROUND(1.0 * SUM(traffic_volume > 5500) / COUNT(*), 4)  AS p_congestion,
       ROUND(1.0 * SUM(weather_main = 'Clear') / COUNT(*), 4) AS p_clear,
       ROUND(1.0 * SUM(traffic_volume > 5500 AND weather_main = 'Clear') / COUNT(*), 4) AS p_cong_and_clear
FROM traffic_hourly;

-- Task 3.2a: P(Clear | Congestion) and P(High temp > 292K | Congestion)
SELECT ROUND(1.0 * SUM(weather_main = 'Clear') / COUNT(*), 4) AS p_clear_given_cong,
       ROUND(1.0 * SUM(temp > 292) / COUNT(*), 4)             AS p_hightemp_given_cong
FROM traffic_hourly
WHERE traffic_volume > 5500;

-- Task 3.2b: independence test for Congestion and Clear weather
SELECT ROUND(1.0 * SUM(traffic_volume > 5500 AND weather_main = 'Clear') / COUNT(*), 4) AS p_joint,
       ROUND((1.0 * SUM(traffic_volume > 5500) / COUNT(*)) *
             (1.0 * SUM(weather_main = 'Clear') / COUNT(*)), 4)                          AS p_product
FROM traffic_hourly;

-- Task 3.2c: congestion counts in Clear vs Clouds
SELECT weather_main,
       SUM(traffic_volume > 5500)  AS congested,
       SUM(traffic_volume <= 5500) AS not_congested
FROM traffic_hourly
WHERE weather_main IN ('Clear', 'Clouds')
GROUP BY weather_main;

-- Task 3.2d: odds ratio of congestion, Clear vs Clouds
SELECT ROUND(
  (1.0 * SUM(weather_main = 'Clear'  AND traffic_volume > 5500) /
         SUM(weather_main = 'Clear'  AND traffic_volume <= 5500)) /
  (1.0 * SUM(weather_main = 'Clouds' AND traffic_volume > 5500) /
         SUM(weather_main = 'Clouds' AND traffic_volume <= 5500)), 4) AS odds_ratio
FROM traffic_hourly;