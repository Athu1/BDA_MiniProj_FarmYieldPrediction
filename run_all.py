"""
Run the whole Big Data pipeline end to end.

    python run_all.py                 generate data + every Spark step (~15 min on a 4-core laptop)
    python run_all.py --fast          train the MLlib model on a 25 % sample (much quicker)
    python run_all.py --skip-generate reuse the existing data/raw files

Steps: 0 generate raw data -> 1 ingest (bronze/silver) -> 2 features (gold) -> 3 Spark SQL analytics
       -> 4 MLlib training -> 5 streaming replay -> 6 benchmark -> 7 serving model for the dashboard
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--farms", type=int, default=200_000)
    a = ap.parse_args()
    t0 = time.time()

    if not a.skip_generate:
        step("0 / 7  Generate simulated raw data (landing zone)")
        subprocess.run([sys.executable, str(ROOT / "data_generator.py"), "--farms", str(a.farms)], check=True)

    from bda import analytics, benchmark, features, ingest, serving_model, streaming, train_model
    from bda.config import get_spark
    spark = get_spark("MaharashtraCropBDA", memory="5g")

    step("1 / 7  Ingest: raw CSV + JSON -> Bronze -> Silver Parquet (+ data-quality report)")
    ingest.run(spark)
    step("2 / 7  Feature engineering: Silver -> Gold (window functions, 5-table join)")
    features.run(spark)
    step("3 / 7  Spark SQL analytics")
    analytics.run(spark)
    step("4 / 7  Spark MLlib: linear baseline + tuned gradient-boosted trees (temporal split)")
    train_model.run(spark, fast=a.fast)
    spark.stop()

    step("5 / 7  Structured Streaming: replay 2026 submissions and score them live")
    subprocess.run([sys.executable, str(ROOT / "stream_simulator.py"), "--batches", "30", "--interval", "0"], check=True)
    subprocess.run([sys.executable, "-m", "bda.streaming", "--once", "--reset"], check=True, cwd=ROOT)
    step("6 / 7  Benchmark: pandas vs Spark, CSV vs Parquet")
    subprocess.run([sys.executable, "-m", "bda.benchmark"], check=True, cwd=ROOT)
    step("7 / 7  Serving model for the dashboard")
    serving_model.run()
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} minutes. Launch the dashboard: python dashboard/app.py")


if __name__ == "__main__":
    main()
