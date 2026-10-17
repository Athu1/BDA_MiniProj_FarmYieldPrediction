"""
Step 2 - Feature engineering: Silver tables -> Gold analytical table (one row per farm-season).

Weather: ~615k daily rows are reduced to one row per weather station per season, using
window functions:
  * monsoon onset   = first day whose 3-day rainfall total reaches 40 mm   (rolling window)
  * longest dry spell between onset and 30 Sep (days < 2.5 mm)              (gaps-and-islands)
  * season rainfall (scaled up for missing days), rainy days, mean temperature
Then five tables are joined: farm_seasons + farms + talukas + districts + season weather + soil.
"""
import time

from pyspark.sql import Window
from pyspark.sql import functions as F

from bda.config import GOLD, SILVER, get_spark, p

SEASON_DAYS = 153


def season_weather(spark, years=None):
    w = spark.read.parquet(p(SILVER / "weather_daily"))
    if years is not None:
        w = w.filter(F.col("year").isin(list(years)))                         # partition pruning
    by_stn = Window.partitionBy("station_id", "year").orderBy("obs_date")
    w = (w.withColumn("rain0", F.coalesce("rain_mm", F.lit(0.0)))
         .withColumn("rain_3day", F.sum("rain0").over(by_stn.rowsBetween(0, 2))))
    onset = (w.filter(F.col("rain_3day") >= 40).groupBy("station_id", "year")
             .agg(F.min("obs_date").alias("onset_date")))
    w = w.join(onset, ["station_id", "year"], "left").withColumn(
        "onset_date", F.coalesce("onset_date", F.date_add(F.make_date("year", F.lit(6), F.lit(1)), 45)))

    # gaps-and-islands: consecutive dry days share the same (row_number - dry_row_number) value
    spell = (w.filter((F.col("obs_date") >= F.col("onset_date")) &
                      (F.col("obs_date") <= F.make_date("year", F.lit(9), F.lit(30))))
             .withColumn("is_dry", (F.col("rain0") < 2.5).cast("int"))
             .withColumn("rn", F.row_number().over(by_stn))
             .withColumn("rn_dry", F.row_number().over(Window.partitionBy("station_id", "year", "is_dry").orderBy("obs_date")))
             .filter("is_dry = 1").withColumn("island", F.col("rn") - F.col("rn_dry"))
             .groupBy("station_id", "year", "island").count()
             .groupBy("station_id", "year").agg(F.max("count").alias("longest_dry_spell_days")))

    agg = (w.groupBy("station_id", "taluka_id", "year", "onset_date").agg(
        (F.sum("rain_mm") * SEASON_DAYS / F.count("rain_mm")).alias("season_rain_mm"),
        F.sum((F.col("rain_mm") >= 2.5).cast("int")).alias("rainy_days"),
        F.avg((F.col("tmax_c") + F.col("tmin_c")) / 2).alias("mean_temp_c"),
        F.max("tmax_c").alias("max_temp_c"),
        F.count("rain_mm").alias("observed_days")))
    return (agg.join(spell, ["station_id", "year"], "left")
            .withColumn("longest_dry_spell_days", F.coalesce("longest_dry_spell_days", F.lit(0))))


def build_gold(spark, fs, weather_season):
    farms = spark.read.parquet(p(SILVER / "farms"))
    talukas = spark.read.parquet(p(SILVER / "talukas"))
    districts = spark.read.parquet(p(SILVER / "districts"))
    soil = spark.read.parquet(p(SILVER / "soil_latest")).select(
        "farm_id", "soil_ph", "soil_oc_pct", "soil_n_kg_ha", "soil_p_kg_ha", "soil_k_kg_ha")
    geo = (talukas.join(F.broadcast(districts), "district_code")
           .select("taluka_id", "district_code", "district", "region", "normal_rain_mm"))
    return (fs.join(farms.select("farm_id", "taluka_id", "area_ha"), "farm_id")
            .join(F.broadcast(geo), "taluka_id")
            .join(weather_season.drop("station_id").withColumnRenamed("year", "season_year"),
                  ["taluka_id", "season_year"], "left")
            .join(soil, "farm_id", "left")
            .withColumn("sowing_delay_days", F.datediff("sowing_date", "onset_date"))
            .withColumn("rain_departure_pct", F.round((F.col("season_rain_mm") / F.col("normal_rain_mm") - 1) * 100, 1))
            .withColumn("rain_category", F.when(F.col("rain_departure_pct") < -19, "deficient")
                        .when(F.col("rain_departure_pct") > 19, "excess").otherwise("normal")))


def run(spark=None):
    t0 = time.time()
    spark = spark or get_spark("02-features")
    ws = season_weather(spark)
    ws.write.mode("overwrite").partitionBy("year").parquet(p(GOLD / "season_weather"))
    ws = spark.read.parquet(p(GOLD / "season_weather"))
    print(f"[{time.time()-t0:5.1f}s] season weather: {ws.count():,} station-seasons")
    fs = spark.read.parquet(p(SILVER / "farm_seasons"))
    gold = build_gold(spark, fs, ws)
    gold.write.mode("overwrite").partitionBy("season_year").parquet(p(GOLD / "farm_season_features"))
    n = spark.read.parquet(p(GOLD / "farm_season_features")).count()
    print(f"[{time.time()-t0:5.1f}s] gold farm_season_features: {n:,} rows")


if __name__ == "__main__":
    run()
