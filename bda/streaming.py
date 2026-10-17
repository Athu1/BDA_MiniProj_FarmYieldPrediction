"""
Step 5 - Real-time scoring with Spark Structured Streaming.

New 2026-season farm records arrive as JSON files (from stream_simulator.py). For every micro-batch:
  clean (same rules as batch) -> join static tables (farm, district, soil card, 2026 season weather)
  -> apply the trained MLlib pipeline -> append predictions to the data lake -> refresh dashboard summary.
A second streaming query keeps a windowed count of submissions per region (event-time + watermark).

Usage:  python -m bda.streaming --once          process whatever has arrived, then stop
        python -m bda.streaming                 keep running (start stream_simulator.py in another window)
"""
import argparse
import json
import shutil
import time

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from bda.cleaning import STREAM_SCHEMA, clean_farm_seasons
from bda.config import CHECKPOINTS, MODEL_OUT, OUTPUTS, STREAM_IN, STREAM_OUT, STREAM_YEAR, get_spark, p
from bda.features import build_gold, season_weather


def run(once: bool = False, reset: bool = False, timeout: int | None = None):
    spark = get_spark("05-streaming")
    if reset:
        shutil.rmtree(STREAM_OUT, ignore_errors=True); shutil.rmtree(CHECKPOINTS, ignore_errors=True)
    STREAM_IN.mkdir(parents=True, exist_ok=True)
    model = PipelineModel.load(p(MODEL_OUT / "gbt_pipeline"))
    weather_2026 = season_weather(spark, years=[STREAM_YEAR]).cache()      # static side of the join
    print(f"2026 season weather ready for {weather_2026.count()} stations; waiting for files in {STREAM_IN}")

    stream = (spark.readStream.schema(STREAM_SCHEMA).option("maxFilesPerTrigger", 3)
              .json(p(STREAM_IN))
              .withColumn("event_time", F.to_timestamp("event_time")))
    stats = {"batches": 0, "records": 0}

    def score_batch(batch_df, batch_id):
        if batch_df.isEmpty():
            return
        t = time.time()
        clean = clean_farm_seasons(batch_df, with_yield=False)
        scored = model.transform(build_gold(spark, clean, weather_2026))
        out = scored.select("record_id", "farm_id", "district", "region", "crop", "variety_type", "irrigated",
                            "season_rain_mm", "longest_dry_spell_days", "event_time",
                            F.round("prediction", 3).alias("predicted_yield_t_ha"),
                            F.current_timestamp().alias("scored_at"), F.lit(batch_id).alias("batch_id"))
        out.write.mode("append").parquet(p(STREAM_OUT / "predictions"))
        allp = spark.read.parquet(p(STREAM_OUT / "predictions"))
        (allp.groupBy("district", "crop").agg(F.count("*").alias("records"),
                                              F.round(F.avg("predicted_yield_t_ha"), 3).alias("mean_predicted_yield"))
         .toPandas().to_csv(OUTPUTS / "stream_summary.csv", index=False))
        stats["batches"] += 1
        stats["records"] = allp.count()
        stats.update(last_batch_id=batch_id, last_batch_rows=out.count(),
                     last_batch_seconds=round(time.time() - t, 2), updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(stats, open(OUTPUTS / "stream_status.json", "w"), indent=2)
        print(f"batch {batch_id}: scored {stats['last_batch_rows']} records in {stats['last_batch_seconds']}s "
              f"(total {stats['records']:,})")

    trig = {"availableNow": True} if once else {"processingTime": "5 seconds"}
    q1 = (stream.writeStream.foreachBatch(score_batch).trigger(**trig)
          .option("checkpointLocation", p(CHECKPOINTS / "scoring")).start())
    q2 = (stream.withWatermark("event_time", "2 minutes")
          .groupBy(F.window("event_time", "1 minute"), "crop").count()
          .writeStream.outputMode("update").format("memory").queryName("submissions_per_minute")
          .trigger(**trig).option("checkpointLocation", p(CHECKPOINTS / "windowed")).start())
    if once:
        q1.awaitTermination(); q2.awaitTermination()
    else:
        try:
            q1.awaitTermination(timeout)
        except KeyboardInterrupt:
            pass
        q1.stop(); q2.stop()
    # update mode re-emits a window each time its count changes -> keep the latest (largest) count
    spark.sql("""SELECT date_format(window.start, 'yyyy-MM-dd HH:mm') AS minute, crop, MAX(count) AS submissions
                 FROM submissions_per_minute GROUP BY 1, 2 ORDER BY 1, 2""") \
        .toPandas().to_csv(OUTPUTS / "stream_windowed_counts.csv", index=False)
    print(f"stream finished: {stats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process available files then stop")
    ap.add_argument("--reset", action="store_true", help="clear previous stream output and checkpoints")
    ap.add_argument("--timeout", type=int, default=None, help="seconds to run in continuous mode")
    a = ap.parse_args()
    run(a.once, a.reset, a.timeout)
