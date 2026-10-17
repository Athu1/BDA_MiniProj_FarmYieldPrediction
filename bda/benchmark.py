"""
Step 6 - Benchmark: single-machine pandas vs Spark, CSV vs Parquet, with and without partition pruning.

This is an honest test: on ~2 M rows that fit in RAM, pandas is often as fast or faster than Spark,
because Spark pays for planning, task scheduling and JVM<->Python hand-offs. Spark's value is that the
SAME code keeps working when the data no longer fits on one machine, and that columnar,
partitioned storage reads far less data. The numbers in outputs/benchmark.csv show both effects.
"""
import glob
import time
from pathlib import Path

import pandas as pd

from bda.config import GOLD, OUTPUTS, RAW, SILVER, get_spark, p


def timed(fn):
    t = time.time(); out = fn(); return out, round(time.time() - t, 2)


def folder_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6, 1)


def run(spark=None):
    spark = spark or get_spark("06-benchmark")
    from pyspark.sql import functions as F
    rows = []
    csvs = sorted(glob.glob(str(RAW / "farm_seasons" / "*.csv")))
    spark.read.parquet(p(GOLD / "farm_season_features")).count()          # warm up the JVM

    pdf, t = timed(lambda: pd.concat(pd.read_csv(f) for f in csvs))
    _, ts = timed(lambda: spark.read.csv(p(RAW / "farm_seasons"), header=True, inferSchema=False).count())
    rows.append(("Read 2 M-row raw CSV (10 files)", t, ts, f"{len(pdf):,} rows"))
    rows.append(("Storage size: raw CSV vs Silver Parquet (MB)", folder_mb(RAW / "farm_seasons"),
                 folder_mb(SILVER / "farm_seasons"), "snappy-compressed columnar"))
    del pdf

    gold_pd, t = timed(lambda: pd.read_parquet(GOLD / "farm_season_features"))
    rows.append(("Load Gold table fully into memory", t, None,
                 f"pandas uses {gold_pd.memory_usage(deep=True).sum() / 1e6:,.0f} MB RAM vs "
                 f"{folder_mb(GOLD / 'farm_season_features')} MB on disk"))

    g = spark.read.parquet(p(GOLD / "farm_season_features"))
    _, t = timed(lambda: gold_pd.groupby(["district", "crop", "season_year"], observed=True).yield_t_ha.mean())
    _, ts = timed(lambda: g.groupBy("district", "crop", "season_year").agg(F.avg("yield_t_ha")).collect())
    rows.append(("Group-by mean: district x crop x year", t, ts, "pandas excludes its load time"))

    _, ts_all = timed(lambda: g.groupBy("crop").agg(F.avg("yield_t_ha")).collect())
    _, ts_one = timed(lambda: g.filter("season_year = 2024").groupBy("crop").agg(F.avg("yield_t_ha")).collect())
    rows.append(("Spark: all 10 seasons vs one season (partition pruning)", ts_all, ts_one,
                 "left = full scan, right = only season_year=2024 folder read"))

    fs_pd = pd.read_parquet(SILVER / "farm_seasons", columns=["farm_id", "crop", "yield_t_ha"])
    farms_pd, tal_pd = pd.read_parquet(SILVER / "farms"), pd.read_parquet(SILVER / "talukas")
    _, t = timed(lambda: fs_pd.merge(farms_pd, on="farm_id").merge(tal_pd, on="taluka_id")
                 .groupby(["district_code", "crop"], observed=True).yield_t_ha.mean())
    fs_s = spark.read.parquet(p(SILVER / "farm_seasons"))
    _, ts = timed(lambda: fs_s.join(spark.read.parquet(p(SILVER / "farms")), "farm_id")
                  .join(F.broadcast(spark.read.parquet(p(SILVER / "talukas"))), "taluka_id")
                  .groupBy("district_code", "crop").agg(F.avg("yield_t_ha")).collect())
    rows.append(("Join 2 M x 200 k x 367 rows, then aggregate", t, ts, "Spark broadcasts the small table"))

    out = pd.DataFrame(rows, columns=["test", "pandas_or_left", "spark_or_right", "note"])
    out.to_csv(OUTPUTS / "benchmark.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
