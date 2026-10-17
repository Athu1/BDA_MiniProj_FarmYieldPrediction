"""
Step 7 - Lightweight serving model for the desktop dashboard.

The Spark MLlib pipeline is the system of record (batch + streaming scoring), but starting a JVM
for every click in a desktop app is slow. A common production pattern is to export a compact
model for interactive use. Here a scikit-learn HistGradientBoosting model is trained on a
300 k-row sample of the Gold table, with the same features and the same temporal split, and its
accuracy is reported next to the Spark model.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from bda.config import GOLD, SERVING_OUT, TEST_YEARS
from bda.train_model import CATEGORICAL, NUMERIC, TARGET

FEATURES = NUMERIC + CATEGORICAL


def run():
    cols = FEATURES + [TARGET, "season_year", "district", "normal_rain_mm"]
    df = pd.read_parquet(GOLD / "farm_season_features", columns=cols)
    df["season_year"] = df.season_year.astype(int)
    train = df[~df.season_year.isin(TEST_YEARS)]
    train = train.sample(min(300_000, len(train)), random_state=42)
    test = df[df.season_year.isin(TEST_YEARS)]
    test = test.sample(min(100_000, len(test)), random_state=42)
    cats = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}

    def encode(x):
        x = x[FEATURES].copy()
        for c in CATEGORICAL:
            x[c] = pd.Categorical(x[c], categories=cats[c])
        return x

    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, max_leaf_nodes=63,
                                          min_samples_leaf=40, categorical_features="from_dtype", random_state=42)
    model.fit(encode(train), train[TARGET])
    pred = model.predict(encode(test))
    metrics = {"overall": {"mae": round(mean_absolute_error(test[TARGET], pred), 4),
                           "r2": round(r2_score(test[TARGET], pred), 4)}}
    for crop, idx in test.groupby("crop").groups.items():
        m = test.index.get_indexer(idx)
        metrics[crop] = {"mae": round(mean_absolute_error(test[TARGET].iloc[m], pred[m]), 4),
                         "r2": round(r2_score(test[TARGET].iloc[m], pred[m]), 4)}

    # defaults the GUI uses to pre-fill a district's typical conditions
    by_d = df.groupby(["district", "district_code", "region"]).agg(
        season_rain_mm=("season_rain_mm", "median"), rainy_days=("rainy_days", "median"),
        longest_dry_spell_days=("longest_dry_spell_days", "median"), mean_temp_c=("mean_temp_c", "median"),
        soil_ph=("soil_ph", "median"), soil_oc_pct=("soil_oc_pct", "median"), soil_n_kg_ha=("soil_n_kg_ha", "median"),
        soil_p_kg_ha=("soil_p_kg_ha", "median"), soil_k_kg_ha=("soil_k_kg_ha", "median"),
        normal_rain_mm=("normal_rain_mm", "first")).round(2).reset_index()
    ranges = {c: [float(df[c].quantile(.005)), float(df[c].quantile(.995))] for c in NUMERIC}
    SERVING_OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "categories": cats, "metrics": metrics,
                 "district_defaults": by_d, "ranges": ranges}, SERVING_OUT / "serving_model.joblib")
    json.dump(metrics, open(SERVING_OUT / "serving_metrics.json", "w"), indent=2)
    print("serving model (scikit-learn, 300k-row sample):", json.dumps(metrics))


if __name__ == "__main__":
    run()
