"""
Generate a realistic *simulated* paddy-rice yield dataset for Indian smallholder farms.

Why simulate?  Public farm-level yield data with matching soil / weather / input records is
rare.  Instead of a straight-line formula, this generator encodes well-known agronomic
response curves, real-world correlations between inputs, measurement error, unobserved
factors and dirty records, so that the modelling problem behaves like real field data.

IMPORTANT: the output is still synthetic.  Present it as "simulated from agronomic
response curves", never as surveyed farm data.

Run:  python generate_data.py          -> writes data/crop_yield_data.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_FARMS = 3000
OUT = Path(__file__).resolve().parent / "data" / "crop_yield_data.csv"# ---------------------------------------------------------------- agronomic parameters
Y_MAX = 8.0              # attainable paddy yield under ideal conditions, t/ha
IRRIGATION_MM = 450      # extra effective water from irrigation, mm
WATER_HALF_MM = 1000     # effective water at which the water response is 50 %
WATER_SLOPE_MM = 170     # steepness of the water (drought) response
FLOOD_START_MM = 2300    # waterlogging losses start above this effective water
TEMP_OPT_C = 27.0        # optimum mean season temperature for rice
TEMP_WIDTH_C = 6.0       # tolerance width of the temperature response
PH_OPT = 6.5             # optimum soil pH for rice
PH_WIDTH = 2.3           # tolerance width of the pH response
N_FROM_OC = 40           # kg N/ha supplied by each 1 % soil organic carbon
N_RESPONSE_KG = 80       # Mitscherlich rate constant for nitrogen
N_ZERO_SHARE = 0.55      # yield share lost with no nitrogen at all
LODGING_START_KG = 170   # nitrogen above this causes lodging losses
LODGING_RATE = 0.0012    # fractional loss per extra kg N/ha
FARM_SIZE_EFFECT = 0.03  # inverse farm-size / productivity relationship
NOISE_SD = 0.12          # unobserved factors (variety, pests, timing), log scale
SHOCK_RATE = 0.04        # share of farms hit by pest / flood / hail events
SHOCK_LOSS = (0.35, 0.65)  # fraction of yield kept after a shock


def generate(seed: int = SEED, n: int = N_FARMS, return_truth: bool = False):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------ true conditions (correlated, skewed)
    rain = np.clip(rng.lognormal(np.log(1250), 0.35, n), 350, 3200)        # Jun-Oct, mm
    irrigated = (rng.random(n) < np.clip(0.75 - (rain - 1250) / 2500, 0.3, 0.9)).astype(int)
    temp = np.clip(27.5 - 0.0015 * (rain - 1250) + rng.normal(0, 2.0, n), 21, 34)
    sun = np.clip(7.2 - 0.0013 * (rain - 1250) + rng.normal(0, 0.9, n), 3.5, 10)  # cloudier when wetter
    ph = np.clip(rng.normal(6.8, 0.9, n), 4.5, 9.0)
    oc = np.clip(rng.lognormal(np.log(0.55), 0.4, n), 0.1, 1.6)            # organic carbon %
    nitrogen = np.where(rng.random(n) < 0.12, 0.0,
                        rng.normal(85 + 30 * irrigated, 50, n))           # irrigated farms use more N
    nitrogen = np.clip(nitrogen, 0, 260)
    size = np.clip(rng.lognormal(np.log(1.4), 0.9, n), 0.2, 40)            # most farms < 2 ha

    # ------------------------------------------------ agronomic response curves (each 0..1)
    water = rain + IRRIGATION_MM * irrigated
    f_water = 1 / (1 + np.exp(-(water - WATER_HALF_MM) / WATER_SLOPE_MM))
    f_water *= 1 - 0.25 * np.clip((water - FLOOD_START_MM) / 700, 0, 1)
    f_temp = np.exp(-(((temp - TEMP_OPT_C) / TEMP_WIDTH_C) ** 2))
    f_sun = 0.70 + 0.30 * np.clip((sun - 3.5) / 5.0, 0, 1)
    f_ph = np.exp(-(((ph - PH_OPT) / PH_WIDTH) ** 2))
    n_eff = nitrogen + N_FROM_OC * oc
    f_n = 1 - N_ZERO_SHARE * np.exp(-n_eff / N_RESPONSE_KG)
    f_n *= 1 - LODGING_RATE * np.clip(nitrogen - LODGING_START_KG, 0, None)
    f_size = 1 - FARM_SIZE_EFFECT * np.log(size / 1.4)

    y = Y_MAX * f_water * f_temp * f_sun * f_ph * f_n * f_size
    y_signal = y.copy()                                                     # noiseless "truth"
    y *= np.exp(rng.normal(0, NOISE_SD, n))                                 # unobserved factors
    shock = rng.random(n) < SHOCK_RATE
    y[shock] *= rng.uniform(*SHOCK_LOSS, shock.sum())                       # pests / floods
    y *= np.exp(rng.normal(0, 0.03, n))                                     # weighing error

    # ------------------------------------------------ what is actually *recorded*
    df = pd.DataFrame({
        "rainfall_mm": np.round(rain * np.exp(rng.normal(0, 0.08, n))),     # district gauge, not farm
        "avg_temp_c": np.round(temp + rng.normal(0, 0.4, n), 1),
        "sunlight_hours": np.round(sun + rng.normal(0, 0.3, n), 1),
        "soil_ph": np.round(ph + rng.normal(0, 0.15, n), 1),                 # Soil Health Card test
        "organic_carbon_pct": np.round(oc * np.exp(rng.normal(0, 0.1, n)), 2),
        "nitrogen_kg_ha": np.round(nitrogen * np.exp(rng.normal(0, 0.12, n))),  # farmer recall
        "irrigated": irrigated,
        "farm_size_ha": np.round(size, 2),
        "yield_t_ha": np.round(y, 2),
    })

    # ------------------------------------------------ real-world data problems
    df.loc[rng.random(n) < 0.03, "nitrogen_kg_ha"] = np.nan                # not reported
    df.loc[rng.random(n) < 0.05, "soil_ph"] = np.nan                       # no soil test
    df.loc[rng.random(n) < 0.05, "organic_carbon_pct"] = np.nan
    df.loc[rng.random(n) < 0.01, "sunlight_hours"] = np.nan                # station gap
    bad = rng.choice(n, 12, replace=False)                                 # data-entry errors
    df.loc[bad[:4], "rainfall_mm"] = (df.loc[bad[:4], "rainfall_mm"] / 10).round()  # entered in cm
    df.loc[bad[4:7], "soil_ph"] = 0.0
    df.loc[bad[7:10], "yield_t_ha"] = (df.loc[bad[7:10], "yield_t_ha"] * 1000).round()  # kg not t
    df.loc[bad[10:], "nitrogen_kg_ha"] = -df.loc[bad[10:], "nitrogen_kg_ha"].fillna(50)
    dupes = df.sample(20, random_state=seed)                               # double entry
    df = pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)
    if return_truth:
        return df, y_signal, y
    return df


if __name__ == "__main__":
    data = generate()
    OUT.parent.mkdir(exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"Wrote {len(data):,} rows to {OUT}")
    print(data.describe().round(2).T)
