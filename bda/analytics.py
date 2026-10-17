"""
Step 3 - Descriptive and diagnostic analytics with Spark SQL over ~2 M farm-seasons.

Every query is plain SQL against temporary views, so it reads like a data-warehouse report.
Results (small tables) are written to outputs/analytics/*.csv for the dashboard and the report.
"""
import time

from bda.config import ANALYTICS_OUT, GOLD, get_spark, p

QUERIES = {
    # 1. state-level trend per crop with year-on-year change (LAG window function)
    "state_crop_trend": """
        SELECT crop, season_year,
               COUNT(*)                             AS farm_records,
               ROUND(AVG(yield_t_ha), 3)            AS mean_yield,
               ROUND(PERCENTILE_APPROX(yield_t_ha, 0.5), 3) AS median_yield,
               ROUND(AVG(season_rain_mm), 0)        AS mean_rain_mm,
               ROUND(100 * (AVG(yield_t_ha) / LAG(AVG(yield_t_ha)) OVER (PARTITION BY crop ORDER BY season_year) - 1), 1)
                                                    AS yoy_change_pct
        FROM gold GROUP BY crop, season_year ORDER BY crop, season_year""",

    # 2. district x crop x year cube used by the dashboard
    "district_crop_year": """
        SELECT region, district, district_code, crop, season_year,
               COUNT(*) AS farm_records, ROUND(AVG(yield_t_ha), 3) AS mean_yield,
               ROUND(AVG(season_rain_mm), 0) AS mean_rain_mm, ROUND(AVG(rain_departure_pct), 1) AS rain_departure_pct,
               ROUND(AVG(longest_dry_spell_days), 1) AS mean_dry_spell_days,
               ROUND(AVG(irrigated) * 100, 1) AS irrigated_pct
        FROM gold GROUP BY region, district, district_code, crop, season_year""",

    # 3. rainfall departure (IMD-style categories) vs yield, irrigated vs rain-fed
    "rainfall_impact": """
        SELECT crop, rain_category, irrigated, COUNT(*) AS farm_records,
               ROUND(AVG(yield_t_ha), 3) AS mean_yield
        FROM gold GROUP BY crop, rain_category, irrigated ORDER BY crop, rain_category, irrigated""",

    # 4. dry-spell length buckets: the hidden cost of monsoon breaks
    "dry_spell_impact": """
        SELECT crop,
               CASE WHEN longest_dry_spell_days < 10 THEN '0-9'  WHEN longest_dry_spell_days < 15 THEN '10-14'
                    WHEN longest_dry_spell_days < 20 THEN '15-19' WHEN longest_dry_spell_days < 25 THEN '20-24'
                    ELSE '25+' END AS dry_spell_bucket,
               irrigated, COUNT(*) AS farm_records, ROUND(AVG(yield_t_ha), 3) AS mean_yield
        FROM gold GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""",

    # 5. nitrogen response curve per crop (diminishing returns)
    "nitrogen_response": """
        SELECT crop, CAST(FLOOR(nitrogen_kg_ha / 20) * 20 AS INT) AS n_bucket_kg_ha,
               COUNT(*) AS farm_records, ROUND(AVG(yield_t_ha), 3) AS mean_yield
        FROM gold WHERE nitrogen_kg_ha IS NOT NULL GROUP BY 1, 2 HAVING COUNT(*) > 200 ORDER BY 1, 2""",

    # 6. soil pH response per crop (Soil Health Card join)
    "soil_ph_response": """
        SELECT crop, ROUND(soil_ph * 2) / 2 AS ph_bucket, COUNT(*) AS farm_records, ROUND(AVG(yield_t_ha), 3) AS mean_yield
        FROM gold WHERE soil_ph IS NOT NULL GROUP BY 1, 2 HAVING COUNT(*) > 200 ORDER BY 1, 2""",

    # 7. district ranking with yield stability (coefficient of variation across years) and RANK()
    "district_ranking": """
        WITH dy AS (SELECT district, region, crop, season_year, AVG(yield_t_ha) AS y, COUNT(*) AS n
                    FROM gold GROUP BY 1, 2, 3, 4)
        SELECT district, region, crop, ROUND(AVG(y), 3) AS mean_yield,
               ROUND(STDDEV(y) / AVG(y) * 100, 1) AS cv_pct, SUM(n) AS farm_records,
               RANK() OVER (PARTITION BY crop ORDER BY AVG(y) DESC) AS yield_rank
        FROM dy GROUP BY 1, 2, 3 HAVING SUM(n) > 5000 ORDER BY crop, yield_rank""",

    # 8. anomaly detection: district-crop-years far below their own 10-year history
    "yield_anomalies": """
        WITH dy AS (SELECT district, district_code, crop, season_year, AVG(yield_t_ha) AS y,
                           AVG(rain_departure_pct) AS rain_dep, COUNT(*) AS n FROM gold GROUP BY 1, 2, 3, 4),
             st AS (SELECT *, AVG(y) OVER (PARTITION BY district, crop) AS hist_mean,
                              STDDEV(y) OVER (PARTITION BY district, crop) AS hist_sd FROM dy)
        SELECT district, district_code, crop, season_year, ROUND(y, 3) AS mean_yield, ROUND(hist_mean, 3) AS hist_mean,
               ROUND((y - hist_mean) / hist_sd, 2) AS z_score, ROUND(rain_dep, 1) AS rain_departure_pct, n AS farm_records,
               CASE WHEN rain_dep < -19 THEN 'drought (rain deficit)'
                    WHEN rain_dep > 40 THEN 'excess rain / flooding'
                    ELSE 'non-weather: pest or disease suspected' END AS likely_cause
        FROM st WHERE n > 300 AND (y - hist_mean) / hist_sd < -1.8 ORDER BY z_score""",

    # 9. technology adoption: variety mix over time
    "variety_adoption": """
        SELECT season_year, crop, variety_type, COUNT(*) AS farm_records,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY season_year, crop), 1) AS share_pct,
               ROUND(AVG(yield_t_ha), 3) AS mean_yield
        FROM gold GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""",

    # 10. panel analysis: how consistent is each farm? (farms in their district's top quartile, year after year)
    "farm_consistency": """
        WITH r AS (SELECT farm_id, crop, season_year,
                          PERCENT_RANK() OVER (PARTITION BY district, crop, season_year ORDER BY yield_t_ha) AS pr
                   FROM gold),
             f AS (SELECT farm_id, COUNT(*) AS seasons, SUM(CASE WHEN pr >= 0.75 THEN 1 ELSE 0 END) AS top_q_seasons
                   FROM r GROUP BY farm_id)
        SELECT top_q_seasons, COUNT(*) AS farms FROM f WHERE seasons = 10 GROUP BY 1 ORDER BY 1""",

    # 11. sowing delay penalty
    "sowing_delay_impact": """
        SELECT crop, irrigated, CAST(LEAST(GREATEST(FLOOR(sowing_delay_days / 5) * 5, -10), 30) AS INT) AS delay_bucket_days,
               COUNT(*) AS farm_records, ROUND(AVG(yield_t_ha), 3) AS mean_yield
        FROM gold GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""",

    # 12. KPI summary
    "kpis": """
        SELECT COUNT(*) AS farm_seasons, COUNT(DISTINCT farm_id) AS farms, COUNT(DISTINCT district) AS districts,
               COUNT(DISTINCT taluka_id) AS talukas, MIN(season_year) AS first_year, MAX(season_year) AS last_year,
               ROUND(AVG(irrigated) * 100, 1) AS irrigated_pct, ROUND(SUM(area_ha) / 10, 0) AS total_area_ha_per_season
        FROM gold""",
}


def run(spark=None):
    t0 = time.time()
    spark = spark or get_spark("03-analytics")
    spark.read.parquet(p(GOLD / "farm_season_features")).createOrReplaceTempView("gold")
    spark.table("gold").cache().count()                     # cache the 2 M rows once, reuse for all queries
    ANALYTICS_OUT.mkdir(parents=True, exist_ok=True)
    timings = []
    for name, sql in QUERIES.items():
        t = time.time()
        pdf = spark.sql(sql).toPandas()
        pdf.to_csv(ANALYTICS_OUT / f"{name}.csv", index=False)
        timings.append((name, len(pdf), round(time.time() - t, 2)))
        print(f"[{time.time()-t0:5.1f}s] {name:22s} -> {len(pdf):6,} rows in {timings[-1][2]:.2f}s")

    spark.catalog.clearCache()
    # show partition pruning: only the 2024 folder is read
    plan = spark.read.parquet(p(GOLD / "farm_season_features")).filter("season_year = 2024") \
        .groupBy("crop").count()._jdf.queryExecution().executedPlan().toString()
    (ANALYTICS_OUT / "query_plan_partition_pruning.txt").write_text(plan)
    import pandas as pd
    pd.DataFrame(timings, columns=["query", "result_rows", "seconds"]).to_csv(ANALYTICS_OUT / "_query_timings.csv", index=False)
    print(f"[{time.time()-t0:5.1f}s] analytics done")


if __name__ == "__main__":
    run()
