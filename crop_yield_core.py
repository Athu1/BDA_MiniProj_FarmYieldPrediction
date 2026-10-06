from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_data.csv"
RESULTS_DIR = BASE_DIR / "crop_yield_results"

TARGET = "yield_t_ha"
# feature -> (label, plausible min, plausible max, GUI step, GUI default)
FEATURE_SPECS = {
    "rainfall_mm": ("Season rainfall (mm)", 250, 4000, 10, 1200),
    "avg_temp_c": ("Mean temperature (°C)", 15, 40, 0.1, 27.0),
    "sunlight_hours": ("Sunshine (hours/day)", 2, 12, 0.1, 7.0),
    "soil_ph": ("Soil pH", 3.5, 10, 0.1, 6.8),
    "organic_carbon_pct": ("Soil organic carbon (%)", 0.05, 3, 0.01, 0.55),
    "nitrogen_kg_ha": ("Nitrogen applied (kg/ha)", 0, 400, 1, 110),
    "irrigated": ("Irrigated (0 = no, 1 = yes)", 0, 1, 1, 1),
    "farm_size_ha": ("Farm size (ha)", 0.1, 100, 0.1, 1.5),
}
FEATURES = list(FEATURE_SPECS)
FEATURE_LABELS = {f: spec[0] for f, spec in FEATURE_SPECS.items()}
TARGET_RANGE = (0.0, 15.0)  # t/ha; anything above is a unit error (kg recorded as t)


def load_data(path: Path = DATA_PATH, verbose: bool = False) -> pd.DataFrame:
    """Read the CSV, enforce schema, remove duplicates and physically impossible rows.

    Missing feature values are KEPT (they are imputed inside each model pipeline, after the
    train/test split, so no information leaks from test rows into training).
    """
    raw = pd.read_csv(path)
    expected = FEATURES + [TARGET]
    if list(raw.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(raw.columns)}")
    df = raw.apply(pd.to_numeric, errors="coerce")

    report = {"rows_read": len(df)}
    df = df.drop_duplicates()
    report["duplicates_removed"] = report["rows_read"] - len(df)

    keep = df[TARGET].between(*TARGET_RANGE)
    for f, (_, lo, hi, _, _) in FEATURE_SPECS.items():
        keep &= df[f].isna() | df[f].between(lo, hi)
    keep &= df["irrigated"].isin([0, 1])
    report["implausible_removed"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    report["rows_kept"] = len(df)
    report["missing_by_feature"] = df[FEATURES].isna().sum()[lambda s: s > 0].to_dict()
    df.attrs["cleaning_report"] = report
    if verbose:
        print("Cleaning report:", report)
    return df


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def _imputer():
    return SimpleImputer(strategy="median")


def build_models() -> dict:
    return {
        "Mean baseline": Pipeline([("impute", _imputer()), ("model", DummyRegressor(strategy="mean"))]),
        "Linear regression": Pipeline([("impute", _imputer()), ("model", LinearRegression())]),
        "Polynomial ridge (deg 2)": Pipeline([
            ("impute", _imputer()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random forest": Pipeline([
            ("impute", _imputer()),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=0.5,
                random_state=SEED, n_jobs=-1)),
        ]),
        "Gradient boosting": HistGradientBoostingRegressor(  # handles missing values natively
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=SEED),
    }


def train_bundle(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=SEED
    )
    cv_split = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in build_models().items():
        cv = cross_validate(model, X_train, y_train, cv=cv_split,
                            scoring=("neg_mean_absolute_error", "r2"), n_jobs=1)
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({
            "Model": name,
            "CV_MAE": float(-cv["test_neg_mean_absolute_error"].mean()),
            "CV_R2_mean": float(cv["test_r2"].mean()),
            "CV_R2_std": float(cv["test_r2"].std()),
            **evaluate(y_test, model.predict(X_test)),
        })

    # select on cross-validation (training data only); the test set is used once, for reporting
    metrics_df = pd.DataFrame(rows).set_index("Model").sort_values("CV_MAE")
    metrics_df = metrics_df[["CV_MAE", "CV_R2_mean", "CV_R2_std", "MAE", "RMSE", "R2"]]
    best_name = metrics_df.index[0]
    best_model = fitted[best_name]
    preds = best_model.predict(X_test)

    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                  random_state=SEED, scoring="neg_mean_absolute_error")
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    lin = fitted["Linear regression"].named_steps["model"]

    return {
        "df": df,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "fitted": fitted,
        "metrics_df": metrics_df,
        "best_name": best_name,
        "best_model": best_model,
        "prediction": preds,
        "bounds": df[FEATURES].agg(["min", "max"]),   # full cleaned data, NaN ignored
        "importance": importance,
        "coefficients": pd.Series(lin.coef_, index=FEATURES),
        "intercept": float(lin.intercept_),
    }


def predict_yield(model, feature_bounds: pd.DataFrame, values: dict) -> float:
    farm = pd.DataFrame([values], columns=FEATURES).apply(pd.to_numeric, errors="raise")
    if not np.isfinite(farm.to_numpy(dtype=float)).all():
        raise ValueError("Every input must be a finite number.")
    if farm.loc[0, "irrigated"] not in (0, 1):
        raise ValueError("irrigated must be 0 or 1.")
    low, high = feature_bounds.loc["min"], feature_bounds.loc["max"]
    outside = (farm.iloc[0] < low) | (farm.iloc[0] > high)
    if outside.any():
        raise ValueError(f"Inputs outside training ranges:\n{feature_bounds.loc[:, outside]}")
    return float(model.predict(farm)[0])


def save_artifacts(bundle: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    bundle["df"].to_csv(RESULTS_DIR / "crop_yield_clean.csv", index=False)
    bundle["metrics_df"].to_csv(RESULTS_DIR / "model_metrics.csv")
    out = bundle["X_test"].copy()
    out["actual_yield"] = bundle["y_test"].to_numpy()
    out["predicted_yield"] = bundle["prediction"]
    out.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    bundle["bounds"].to_csv(RESULTS_DIR / "feature_bounds.csv")
    bundle["importance"].rename("importance_mae").to_csv(RESULTS_DIR / "feature_importance.csv")
    joblib.dump(
        {key: bundle[key] for key in
         ("best_name", "bounds", "metrics_df", "importance", "coefficients", "intercept")}
        | {"model": bundle["best_model"]},
        RESULTS_DIR / "best_model.joblib",
    )


def load_saved_bundle() -> dict | None:
    model_path = RESULTS_DIR / "best_model.joblib"
    if not model_path.exists():
        return None
    payload = joblib.load(model_path)
    needed = {"model", "best_name", "bounds", "metrics_df"}
    if not isinstance(payload, dict) or not needed <= payload.keys():
        return None
    if list(payload["bounds"].columns) != FEATURES:
        return None  # saved with an older feature set -> retrain
    return {"df": load_data(), "best_model": payload["model"], **{k: v for k, v in payload.items() if k != "model"}}
