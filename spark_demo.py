"""
Live demo of Apache Spark for the viva / presentation.

Runs the project's key Big Data operations step by step on the 2-million-row data lake while you show
Spark's own monitoring dashboard, the Spark Web UI, at http://localhost:4040.

    python spark_demo.py           guided: press Enter between steps, then an interactive SQL prompt
    python spark_demo.py --auto    run all steps without pausing (for testing)

Requires the data lake: run `python run_all.py` (or at least data_generator + ingest + features) first.
Spark keeps running (and the Web UI stays open) until you type `exit` at the SQL prompt.
"""
import argparse
import shutil
import time

from bda.config import DATA, GOLD, MODEL_OUT, SILVER, STREAM_IN, get_spark, p

BAR = "=" * 78


def main(auto: bool):
    spark = get_spark("FieldYield-LIVE-DEMO", memory="4g")
    sc = spark.sparkContext
    from pyspark.sql import functions as F

    def step(n, title, explain):
        print(f"\n{BAR}\nSTEP {n}: {title}\n{BAR}\n{explain}")
        if not auto:
            input("\n   [Enter] to run this step ... ")
        sc.setJobGroup(f"step{n}", f"Step {n}: {title}")          # readable names in the Web UI

    print(f"\n{BAR}\n  Spark {spark.version} is running on {sc.defaultParallelism} cores (master = {sc.master})")
    port = (sc.uiWebUrl or ":4040").rsplit(":", 1)[-1]
    print(f"  >>> OPEN THE SPARK WEB UI IN YOUR BROWSER:  http://localhost:{port}")
    print(f"{BAR}")

    # 1 -------------------------------------------------------------- read the lake
    step(1, "Read the Gold data lake (Parquet, partitioned by season)",
         "Spark reads 10 season folders lazily: nothing runs until an action such as count().\n"
         "Web UI: Jobs tab shows one job; Stages tab shows tasks split across the cores.")
    gold = spark.read.parquet(p(GOLD / "farm_season_features"))
    gold.createOrReplaceTempView("gold")
    t = time.time(); n = gold.count()
    print(f"   {n:,} farm-season rows, {len(gold.columns)} columns, counted in {time.time()-t:.1f}s")
    print("   partition folders:", ", ".join(sorted(x.name for x in (GOLD / "farm_season_features").iterdir() if x.is_dir()))[:200])

    # 2 -------------------------------------------------------------- SQL with shuffle
    step(2, "Spark SQL: district x crop yield ranking (GROUP BY + window RANK)",
         "A GROUP BY needs a SHUFFLE (rows with the same key move to the same partition).\n"
         "Web UI: SQL / DataFrame tab -> click the query -> see the plan with 'Exchange' (shuffle).")
    spark.sql("""
        SELECT * FROM (
          SELECT crop, district, ROUND(AVG(yield_t_ha), 3) AS mean_yield_t_ha, COUNT(*) AS farm_seasons,
                 RANK() OVER (PARTITION BY crop ORDER BY AVG(yield_t_ha) DESC) AS rank_in_crop
          FROM gold GROUP BY crop, district) WHERE rank_in_crop <= 3 ORDER BY crop, rank_in_crop""").show(12, False)

    # 3 -------------------------------------------------------------- partition pruning
    step(3, "Partition pruning: query one season vs all ten",
         "Filtering on the partition column lets Spark skip 9 of 10 folders.\n"
         "Web UI: SQL tab -> the 2024 query's scan node shows 'number of files read' much smaller.")
    t = time.time(); spark.sql("SELECT crop, AVG(yield_t_ha) FROM gold GROUP BY crop").collect(); full = time.time() - t
    t = time.time()
    spark.sql("SELECT crop, ROUND(AVG(yield_t_ha),3) AS mean_2024 FROM gold WHERE season_year = 2024 GROUP BY crop").show()
    print(f"   all seasons: {full:.2f}s   |   season 2024 only: {time.time()-t:.2f}s")
    spark.sql("SELECT crop FROM gold WHERE season_year = 2024").explain()

    # 4 -------------------------------------------------------------- broadcast join
    step(4, "Join 2 M farm-seasons with farms and talukas (broadcast join)",
         "The small talukas table (367 rows) is copied to every core instead of shuffling 2 M rows.\n"
         "Web UI: SQL tab -> look for 'BroadcastHashJoin' and 'BroadcastExchange'.")
    fs = spark.read.parquet(p(SILVER / "farm_seasons"))
    farms, talukas = spark.read.parquet(p(SILVER / "farms")), spark.read.parquet(p(SILVER / "talukas"))
    (fs.join(farms, "farm_id").join(F.broadcast(talukas), "taluka_id")
       .groupBy("district_code").agg(F.count("*").alias("records"), F.round(F.avg("yield_t_ha"), 3).alias("mean_yield"))
       .orderBy(F.desc("records")).show(8))

    # 5 -------------------------------------------------------------- caching
    step(5, "Cache the Gold table in memory",
         "Repeated queries reuse the in-memory copy.\nWeb UI: Storage tab shows the cached table, its size and partitions.")
    gold.cache(); t = time.time(); gold.count(); first = time.time() - t
    t = time.time(); gold.groupBy("rain_category").count().show(); second = time.time() - t
    print(f"   first pass (reads Parquet + fills cache): {first:.2f}s   |   next query from cache: {second:.2f}s")

    # 6 -------------------------------------------------------------- window functions on weather
    step(6, "Window functions on daily weather: monsoon onset per station",
         "A rolling 3-day rainfall sum over each station's time series (rowsBetween).\n"
         "Web UI: Stages tab shows the sort needed for the window.")
    from pyspark.sql import Window
    w = spark.read.parquet(p(SILVER / "weather_daily")).filter("year = 2025")
    win = Window.partitionBy("station_id").orderBy("obs_date").rowsBetween(0, 2)
    (w.withColumn("rain_3day", F.sum(F.coalesce("rain_mm", F.lit(0.0))).over(win)).filter("rain_3day >= 40")
       .groupBy("station_id").agg(F.min("obs_date").alias("onset_2025")).orderBy("onset_2025").show(5))

    # 7 -------------------------------------------------------------- MLlib
    step(7, "Spark MLlib: score 2025 farms with the trained gradient-boosted-tree pipeline",
         "The saved PipelineModel (indexers, imputer, assembler, GBT) is applied to 200 k rows in parallel.\n"
         "Web UI: Jobs tab shows the scoring and evaluation jobs.")
    try:
        from pyspark.ml import PipelineModel
        from pyspark.ml.evaluation import RegressionEvaluator
        model = PipelineModel.load(p(MODEL_OUT / "gbt_pipeline"))
        pred = model.transform(gold.filter("season_year = 2025"))
        pred.select("district", "crop", "yield_t_ha", F.round("prediction", 2).alias("predicted")).show(5)
        for crop in ("paddy", "soybean", "cotton", "tur"):
            r2 = RegressionEvaluator(labelCol="yield_t_ha", metricName="r2").evaluate(pred.filter(F.col("crop") == crop))
            print(f"   2025 {crop:8s} R2 = {r2:.3f}")
    except Exception as e:                                         # model not trained yet
        print("   (skipped: train the model first with python -m bda.train_model)", e)

    # 8 -------------------------------------------------------------- Structured Streaming
    step(8, "Structured Streaming: replay live 2026 submissions in micro-batches",
         "Spark treats the folder of arriving JSON files as an unbounded table.\n"
         "Web UI: the 'Structured Streaming' tab appears, with input rate and batch duration charts.")
    if STREAM_IN.exists() and any(STREAM_IN.glob("*.json")):
        from bda.cleaning import STREAM_SCHEMA
        ck = DATA / "stream" / "demo_checkpoint"; shutil.rmtree(ck, ignore_errors=True)
        q = (spark.readStream.schema(STREAM_SCHEMA).option("maxFilesPerTrigger", 2).json(p(STREAM_IN))
             .groupBy("crop").count().writeStream.outputMode("complete").format("memory")
             .queryName("live_counts").option("checkpointLocation", p(ck)).trigger(availableNow=True).start())
        q.awaitTermination()
        print(f"   processed {len(q.recentProgress)} micro-batches")
        spark.sql("SELECT * FROM live_counts ORDER BY count DESC").show()
    else:
        print("   (skipped: run python stream_simulator.py first)")

    # interactive SQL ------------------------------------------------
    print(f"\n{BAR}\nDEMO STEPS DONE. Spark and the Web UI stay open: http://localhost:{port}")
    print("Type any Spark SQL against the view `gold` (or `exit`). Try:")
    print("   SELECT region, crop, ROUND(AVG(yield_t_ha),2) y FROM gold GROUP BY region, crop ORDER BY region, y DESC")
    print(f"{BAR}")
    if auto:
        spark.sql("SELECT region, crop, ROUND(AVG(yield_t_ha),2) y FROM gold GROUP BY region, crop ORDER BY region").show(5)
        return spark
    sc.setJobGroup("interactive", "Interactive SQL")
    while True:
        try:
            q = input("spark-sql> ").strip().rstrip(";")
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q:
            try:
                t = time.time(); spark.sql(q).show(30, False); print(f"({time.time()-t:.2f}s)")
            except Exception as e:
                print("Error:", str(e).split("\n")[0])
    spark.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="run without pauses (testing)")
    ap.add_argument("--hold", type=int, default=0, help="with --auto: keep Spark alive N seconds at the end")
    a = ap.parse_args()
    s = main(a.auto)
    if a.auto and a.hold:
        print(f"holding Spark open for {a.hold}s ..."); time.sleep(a.hold)
