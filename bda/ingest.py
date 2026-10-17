"""
Step 1 - Ingestion: raw landing zone (CSV + nested JSON) -> Bronze -> Silver Parquet data lake.

Bronze = raw data stored as Parquet with lineage columns (which file, when ingested).
Silver = cleaned, typed, de-duplicated, unit-normalised data, partitioned for fast queries.
A data-quality report counts every problem that was found and fixed.
"""
import json
import time

from pyspark.sql import Window
from pyspark.sql import functions as F

from bda.cleaning import FARMS_SCHEMA, FARM_SEASON_SCHEMA, WEATHER_SCHEMA, clean_farm_seasons, normalise_crop
from bda.config import BRONZE, OUTPUTS, RAW, SILVER, get_spark, p


def lineage(df):
    return df.withColumn("_source_file", F.input_file_name()).withColumn("_ingested_at", F.current_timestamp())


def run(spark=None) -> dict:
    t0 = time.time()
    spark = spark or get_spark("01-ingest")
    q = {}

    # ---------------------------------------------------------------- BRONZE (schema-on-read, no cleaning)
    fs_raw = lineage(spark.read.csv(p(RAW / "farm_seasons"), header=True, schema=FARM_SEASON_SCHEMA))
    fs_raw.write.mode("overwrite").partitionBy("season_year").parquet(p(BRONZE / "farm_seasons"))
    w_raw = lineage(spark.read.csv(p(RAW / "weather_daily"), header=True, schema=WEATHER_SCHEMA))
    w_raw.write.mode("overwrite").parquet(p(BRONZE / "weather_daily"))
    soil_raw = lineage(spark.read.json(p(RAW / "soil_health_cards.jsonl")))       # schema inferred from nested JSON
    soil_raw.write.mode("overwrite").parquet(p(BRONZE / "soil_health_cards"))
    for name in ("farms", "talukas", "districts"):
        opts = {"schema": FARMS_SCHEMA} if name == "farms" else {"inferSchema": True}
        spark.read.csv(p(RAW / f"{name}.csv"), header=True, **opts).write.mode("overwrite").parquet(p(BRONZE / name))
    print(f"[{time.time()-t0:5.1f}s] bronze written")

    # ---------------------------------------------------------------- SILVER: farm seasons
    fs = spark.read.parquet(p(BRONZE / "farm_seasons")).cache()
    q["farm_seasons_raw_rows"] = fs.count()
    q["exact_duplicate_records"] = q["farm_seasons_raw_rows"] - fs.dropDuplicates(["record_id"]).count()
    q["crop_name_variants_fixed"] = fs.filter(F.col("crop") != normalise_crop("crop")).count()
    q["dates_in_dd_mm_yyyy"] = fs.filter(F.col("sowing_date").contains("/")).count()
    q["yields_reported_in_quintals"] = fs.filter(F.col("yield_unit") == "q/ha").count()
    q["nitrogen_missing"] = fs.filter(F.col("nitrogen_kg_ha").isNull()).count()
    q["nitrogen_implausible_set_null"] = fs.filter(F.col("nitrogen_kg_ha") > 400).count()
    q["variety_missing"] = fs.filter(F.col("variety_type").isNull()).count()

    silver_fs = clean_farm_seasons(fs.dropDuplicates(["record_id"])).drop("_source_file", "_ingested_at")
    silver_fs.write.mode("overwrite").partitionBy("season_year").parquet(p(SILVER / "farm_seasons"))
    q["farm_seasons_silver_rows"] = spark.read.parquet(p(SILVER / "farm_seasons")).count()
    q["implausible_yield_rows_removed"] = (q["farm_seasons_raw_rows"] - q["exact_duplicate_records"]
                                           - q["farm_seasons_silver_rows"])
    fs.unpersist()
    print(f"[{time.time()-t0:5.1f}s] silver farm_seasons: {q['farm_seasons_silver_rows']:,} rows")

    # ---------------------------------------------------------------- SILVER: weather
    w = spark.read.parquet(p(BRONZE / "weather_daily"))
    q["weather_raw_rows"] = w.count()
    q["weather_sentinel_minus999"] = w.filter(F.col("rain_mm") == -999).count()
    talukas = spark.read.parquet(p(BRONZE / "talukas"))
    silver_w = (w.withColumn("obs_date", F.to_date("obs_date"))
                .withColumn("rain_mm", F.when(F.col("rain_mm").between(0, 500), F.col("rain_mm")))
                .withColumn("tmax_c", F.when(F.col("tmax_c").between(15, 50), F.col("tmax_c")))
                .withColumn("tmin_c", F.when(F.col("tmin_c").between(5, 35), F.col("tmin_c")))
                .withColumn("year", F.year("obs_date"))
                .join(F.broadcast(talukas.select("station_id", "taluka_id")), "station_id", "left")
                .dropDuplicates(["station_id", "obs_date"])
                .drop("_source_file", "_ingested_at"))
    silver_w.write.mode("overwrite").partitionBy("year").parquet(p(SILVER / "weather_daily"))
    expected = talukas.count() * 153 * spark.read.parquet(p(SILVER / "weather_daily")).select("year").distinct().count()
    q["weather_missing_days"] = expected - q["weather_raw_rows"]
    print(f"[{time.time()-t0:5.1f}s] silver weather")

    # ---------------------------------------------------------------- SILVER: soil health cards (nested JSON)
    soil = spark.read.parquet(p(BRONZE / "soil_health_cards"))
    q["soil_cards"] = soil.count()
    latest = Window.partitionBy("farm_id").orderBy(F.col("sample_date").desc())
    silver_soil = (soil.select("farm_id", "card_id", "cycle", F.to_date("sample_date").alias("sample_date"),
                               F.col("results.pH").alias("soil_ph"), F.col("results.OC_pct").alias("soil_oc_pct"),
                               F.col("results.N_kg_ha").cast("double").alias("soil_n_kg_ha"),
                               F.col("results.P_kg_ha").alias("soil_p_kg_ha"),
                               F.col("results.K_kg_ha").cast("double").alias("soil_k_kg_ha"),
                               F.col("results.EC_dS_m").alias("soil_ec_ds_m"))
                   .withColumn("rn", F.row_number().over(latest)).filter("rn = 1").drop("rn"))
    silver_soil.write.mode("overwrite").parquet(p(SILVER / "soil_latest"))
    q["farms_with_soil_card"] = spark.read.parquet(p(SILVER / "soil_latest")).count()
    q["soil_cards_missing_ph"] = soil.filter(F.col("results.pH").isNull()).count()

    for name in ("farms", "talukas", "districts"):
        spark.read.parquet(p(BRONZE / name)).write.mode("overwrite").parquet(p(SILVER / name))
    q["farms"] = spark.read.parquet(p(SILVER / "farms")).count()
    q["seconds"] = round(time.time() - t0, 1)

    OUTPUTS.mkdir(exist_ok=True)
    json.dump(q, open(OUTPUTS / "quality_report.json", "w"), indent=2)
    print(f"[{time.time()-t0:5.1f}s] ingestion done. Quality report:")
    for k, v in q.items():
        print(f"   {k:34s} {v:>12,}" if isinstance(v, int) else f"   {k:34s} {v:>12}")
    return q


if __name__ == "__main__":
    run()
