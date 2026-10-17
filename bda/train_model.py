"""
Step 4 - Distributed machine learning with Spark MLlib.

* Temporal validation: train on seasons 2016-2023, test on the two most recent seasons 2024-2025.
  This mimics real forecasting (no future data leaks into training).
* Hyper-parameters are tuned with TrainValidationSplit on a sample of the training seasons,
  then the chosen model is refit on all ~1.6 M training rows.
* Baseline: linear regression on one-hot encoded features. Main model: gradient-boosted trees.
* Metrics are reported overall AND per crop: crops have very different yield levels, so the
  overall R2 flatters any model that only learns "paddy yields more than tur".

Usage: python -m bda.train_model            (full)
       python -m bda.train_model --fast     (trains on a 25 % sample; for slow laptops)
"""
import argparse
import json
import time

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import functions as F

from bda.config import GOLD, MODEL_OUT, TEST_YEARS, get_spark, p

CATEGORICAL = ["crop", "variety_type", "region", "district_code"]
NUMERIC = ["irrigated", "area_ha", "nitrogen_kg_ha", "sowing_delay_days", "season_rain_mm", "rainy_days",
           "longest_dry_spell_days", "mean_temp_c", "soil_ph", "soil_oc_pct", "soil_n_kg_ha",
           "soil_p_kg_ha", "soil_k_kg_ha"]
TARGET = "yield_t_ha"
SEED = 42


def feature_stages(one_hot: bool):
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    imp = Imputer(strategy="median", inputCols=NUMERIC, outputCols=[f"{c}_imp" for c in NUMERIC])
    num = [f"{c}_imp" for c in NUMERIC]
    if one_hot:      # linear model needs one-hot dummies and scaled inputs
        ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL], outputCols=[f"{c}_ohe" for c in CATEGORICAL])
        asm = VectorAssembler(inputCols=num + [f"{c}_ohe" for c in CATEGORICAL], outputCol="raw_features")
        return idx + [imp, ohe, asm, StandardScaler(inputCol="raw_features", outputCol="features")]
    # trees use the indexed categories directly (StringIndexer metadata marks them as categorical)
    return idx + [imp, VectorAssembler(inputCols=num + [f"{c}_idx" for c in CATEGORICAL], outputCol="features")]


def metrics(pred, by_crop=True):
    ev = {m: RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName=m) for m in ("rmse", "mae", "r2")}
    out = {"overall": {m: round(e.evaluate(pred), 4) for m, e in ev.items()}}
    if by_crop:
        for (crop,) in pred.select("crop").distinct().collect():
            part = pred.filter(F.col("crop") == crop)
            out[crop] = {m: round(e.evaluate(part), 4) for m, e in ev.items()}
    return out


def run(spark=None, fast=False):
    t0 = time.time()
    spark = spark or get_spark("04-train-mllib", memory="5g")
    gold = spark.read.parquet(p(GOLD / "farm_season_features"))
    train = gold.filter(~F.col("season_year").isin(*TEST_YEARS))
    test = gold.filter(F.col("season_year").isin(*TEST_YEARS)).cache()
    if fast:
        train = train.sample(fraction=0.25, seed=SEED)
    train = train.cache()
    n_train, n_test = train.count(), test.count()
    print(f"[{time.time()-t0:5.1f}s] train {n_train:,} rows (2016-2023) | test {n_test:,} rows (2024-2025)")
    results = {"n_train": n_train, "n_test": n_test, "fast_mode": fast}

    # ---------------- baseline: linear regression
    lin = Pipeline(stages=feature_stages(one_hot=True) + [LinearRegression(labelCol=TARGET, regParam=0.01)]).fit(train)
    results["linear_regression"] = metrics(lin.transform(test))
    print(f"[{time.time()-t0:5.1f}s] linear regression: {results['linear_regression']['overall']}")

    # ---------------- tuning on a sample (TrainValidationSplit = one 80/20 split, cheaper than k-fold)
    gbt = GBTRegressor(labelCol=TARGET, featuresCol="features", maxBins=64, subsamplingRate=0.8, seed=SEED)
    pipe = Pipeline(stages=feature_stages(one_hot=False) + [gbt])
    grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [5, 7]).addGrid(gbt.maxIter, [40, 80])
            .addGrid(gbt.stepSize, [0.1]).build())
    tvs = TrainValidationSplit(estimator=pipe, estimatorParamMaps=grid, trainRatio=0.8, seed=SEED,
                               evaluator=RegressionEvaluator(labelCol=TARGET, metricName="rmse"), parallelism=1)
    sample = train.sample(fraction=0.10 if not fast else 0.3, seed=SEED)
    tvs_model = tvs.fit(sample)
    tuning = [{"maxDepth": pm[gbt.maxDepth], "maxIter": pm[gbt.maxIter], "stepSize": pm[gbt.stepSize],
               "val_rmse": round(score, 4)} for pm, score in zip(grid, tvs_model.validationMetrics)]
    best = min(tuning, key=lambda r: r["val_rmse"])
    results["tuning"] = tuning
    results["best_params"] = best
    print(f"[{time.time()-t0:5.1f}s] tuning done, best: {best}")

    # ---------------- final fit on all training rows
    gbt.setMaxDepth(best["maxDepth"]).setMaxIter(best["maxIter"]).setStepSize(best["stepSize"])
    model = Pipeline(stages=feature_stages(one_hot=False) + [gbt]).fit(train)
    pred = model.transform(test).cache()
    results["gbt"] = metrics(pred)
    results["train_seconds_total"] = round(time.time() - t0, 1)
    print(f"[{time.time()-t0:5.1f}s] GBT: {results['gbt']['overall']}")

    names = NUMERIC + CATEGORICAL
    imp = model.stages[-1].featureImportances.toArray()
    results["feature_importance"] = dict(sorted(zip(names, map(float, imp)), key=lambda kv: -kv[1]))

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(p(MODEL_OUT / "gbt_pipeline"))
    json.dump(results, open(MODEL_OUT / "metrics.json", "w"), indent=2)
    (pred.select("season_year", "district", "crop", TARGET, F.round("prediction", 3).alias("prediction"))
     .sample(fraction=20000 / max(n_test, 1), seed=SEED).toPandas()
     .to_csv(MODEL_OUT / "test_predictions_sample.csv", index=False))
    (pred.groupBy("season_year", "district", "crop")
     .agg(F.count("*").alias("farms"), F.round(F.avg(TARGET), 3).alias("actual_mean"),
          F.round(F.avg("prediction"), 3).alias("predicted_mean"))
     .toPandas().to_csv(MODEL_OUT / "district_level_test.csv", index=False))
    for crop, m in results["gbt"].items():
        print(f"   {crop:8s} RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  R2 {m['r2']:.3f}")
    print(f"[{time.time()-t0:5.1f}s] saved model to outputs/model/gbt_pipeline")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    run(fast=ap.parse_args().fast)
