"""
Generate a realistic SIMULATED big-data landing zone for Maharashtra kharif crops (2016-2026).

Tables written to data/raw/ (the "landing zone", deliberately messy like real feeds):
  districts.csv                 34 agricultural districts: region, normal rainfall, irrigation share
  talukas.csv                   ~360 talukas (sub-districts) with weather-station ids
  farms.csv                     200,000 farms: taluka, area, irrigation status (2016)
  soil_health_cards.jsonl       nested JSON soil tests (Soil Health Card style), 1-2 per tested farm
  weather_daily/weather_YYYY.csv  daily rain / max / min temperature per taluka, Jun-Oct 2016-2026
  farm_seasons/survey_YYYY.csv  ~2,000,000 farm-season records 2016-2025 with yield
  stream_source_2026.csv        2026 season submissions, replayed later as a live stream

The data is SIMULATED. District rainfall normals and soil types are approximate and only make
the patterns plausible. Yield follows agronomic response curves per crop with noise, pest
outbreaks and data-quality problems. Present it as simulated data, never as survey data.

Usage:  python data_generator.py              (200,000 farms -> ~2.0 M records, about 2-3 minutes)
        python data_generator.py --farms 50000 (quick test)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
TRUTH = ROOT / "data" / "truth"
SEED = 2026
YEARS = list(range(2016, 2027))            # weather years; 2026 = current (streamed) season
LABEL_YEARS = list(range(2016, 2026))      # farm records with harvested yield
SEASON_DAYS = 153                          # 1 Jun .. 31 Oct
SEP30 = 121                                # day index of 30 Sep

# ------------------------------------------------------------------ reference data (approximate)
REGIONS = {  # crop mix, mean temp, median farm size, onset shift (days), wet-day prob, soil pH, OC %
    "Konkan":              dict(mix=(.92, .05, .00, .03), temp=27.3, size=0.7, onset=-4, wet=.72, ph=5.7, oc=.95),
    "North Maharashtra":   dict(mix=(.15, .30, .40, .15), temp=27.8, size=1.4, onset=1, wet=.50, ph=7.4, oc=.55),
    "Western Maharashtra": dict(mix=(.18, .45, .10, .27), temp=26.2, size=1.1, onset=-1, wet=.52, ph=7.2, oc=.62),
    "Marathwada":          dict(mix=(.02, .48, .32, .18), temp=28.4, size=1.7, onset=2, wet=.42, ph=7.8, oc=.45),
    "Vidarbha":            dict(mix=(.08, .34, .42, .16), temp=29.0, size=1.8, onset=4, wet=.50, ph=7.6, oc=.50),
}
CROPS = ["paddy", "soybean", "cotton", "tur"]
# code, district, region, normal Jun-Oct rainfall (mm), irrigated share, farm-count weight
DISTRICTS = [
    ("THN", "Thane", "Konkan", 2500, .10, 2), ("PLG", "Palghar", "Konkan", 2400, .08, 2.5),
    ("RGD", "Raigad", "Konkan", 3100, .08, 2.5), ("RTG", "Ratnagiri", "Konkan", 3200, .05, 3),
    ("SDG", "Sindhudurg", "Konkan", 3000, .06, 2),
    ("NSK", "Nashik", "North Maharashtra", 1000, .35, 6), ("DHL", "Dhule", "North Maharashtra", 600, .25, 4),
    ("NDB", "Nandurbar", "North Maharashtra", 850, .20, 3), ("JLG", "Jalgaon", "North Maharashtra", 700, .30, 6),
    ("AHM", "Ahilyanagar", "North Maharashtra", 520, .40, 7),
    ("PUN", "Pune", "Western Maharashtra", 850, .40, 5), ("STR", "Satara", "Western Maharashtra", 900, .45, 4),
    ("SNG", "Sangli", "Western Maharashtra", 520, .50, 4), ("KLP", "Kolhapur", "Western Maharashtra", 1700, .55, 4),
    ("SLP", "Solapur", "Western Maharashtra", 500, .40, 6),
    ("CSN", "Chh. Sambhajinagar", "Marathwada", 650, .22, 6), ("JLN", "Jalna", "Marathwada", 650, .20, 5),
    ("BED", "Beed", "Marathwada", 600, .18, 6), ("LTR", "Latur", "Marathwada", 720, .20, 5),
    ("DRS", "Dharashiv", "Marathwada", 650, .20, 5), ("NND", "Nanded", "Marathwada", 900, .20, 6),
    ("PBN", "Parbhani", "Marathwada", 800, .18, 5), ("HGL", "Hingoli", "Marathwada", 850, .15, 4),
    ("NGP", "Nagpur", "Vidarbha", 1050, .25, 5), ("WRD", "Wardha", "Vidarbha", 1000, .20, 4),
    ("BHD", "Bhandara", "Vidarbha", 1300, .40, 3.5), ("GND", "Gondia", "Vidarbha", 1350, .40, 3.5),
    ("CHD", "Chandrapur", "Vidarbha", 1200, .30, 5), ("GDC", "Gadchiroli", "Vidarbha", 1450, .25, 4),
    ("AMR", "Amravati", "Vidarbha", 800, .18, 6), ("AKL", "Akola", "Vidarbha", 750, .15, 4.5),
    ("WSM", "Washim", "Vidarbha", 850, .15, 3.5), ("YTL", "Yavatmal", "Vidarbha", 900, .12, 7),
    ("BLD", "Buldhana", "Vidarbha", 700, .15, 5.5),
]
MIX_OVERRIDE = {  # paddy belts and dry districts differ from their region
    "BHD": (.62, .12, .14, .12), "GND": (.65, .10, .13, .12), "GDC": (.62, .12, .14, .12), "CHD": (.45, .20, .22, .13),
    "KLP": (.35, .45, .10, .10), "PUN": (.25, .45, .05, .25), "STR": (.22, .48, .05, .25), "NSK": (.20, .32, .30, .18),
    "NDB": (.20, .25, .40, .15), "SLP": (.00, .45, .15, .40), "SNG": (.02, .50, .10, .38),
    "AHM": (.03, .35, .30, .32), "DHL": (.03, .30, .47, .20), "JLG": (.02, .30, .50, .18),
}
PH_OVERRIDE = {"BHD": 6.6, "GND": 6.5, "GDC": 6.4, "CHD": 6.9, "KLP": 6.6, "NSK": 7.0}

# per-crop agronomy: attainable yield t/ha, water half-response & slope (mm), flood start, flood max loss,
# temperature optimum & width, pH optimum & width, N zero-share & rate, dry-spell sensitivity, typical N use
CROP_P = pd.DataFrame({
    "ymax":      [6.0, 2.0, 1.6, 1.1],
    "w_half":    [1100, 600, 550, 500], "w_slope": [180, 120, 120, 110],
    "fl_start":  [3300, 1500, 1600, 1400], "fl_max": [.15, .35, .35, .40],
    "t_opt":     [27, 26, 28, 27], "t_w": [6, 6, 7, 7],
    "ph_opt":    [6.0, 6.8, 7.5, 7.0], "ph_w": [2.3, 2.3, 2.6, 2.5],
    "n_zero":    [.50, .15, .40, .10], "n_rate": [80, 40, 70, 30],
    "dry_k":     [.020, .015, .012, .010], "n_use": [90, 30, 110, 25],
}, index=CROPS)
VARIETY_MULT = {"local": .82, "improved": 1.0, "hybrid": 1.12}


def _logistic(x, half, slope):
    return 1 / (1 + np.exp(-(x - half) / slope))


# ------------------------------------------------------------------ weather
def make_weather(rng, talukas, districts):
    """Daily rain/temperature for every taluka x year (Markov-chain rain occurrence, gamma amounts)."""
    nT, nY, D = len(talukas), len(YEARS), SEASON_DAYS
    dist = districts.set_index("district_code").loc[talukas.district_code]
    reg = dist.region.map(REGIONS)
    year_anom = rng.normal(0, .16, nY)                                  # state-wide good/bad monsoon
    dist_year = rng.normal(0, .12, (len(districts), nY))
    d_idx = districts.reset_index().set_index("district_code").loc[talukas.district_code, "index"].to_numpy()
    season_total = (dist.normal_rain_mm.to_numpy()[:, None] * talukas.rain_factor.to_numpy()[:, None]
                    * np.exp(year_anom[None, :] + dist_year[d_idx] + rng.normal(0, .07, (nT, nY))))
    onset_true = np.clip(9 + reg.map(lambda r: r["onset"]).to_numpy()[:, None]
                         + rng.normal(0, 4, (1, nY)) + rng.normal(0, 3, (nT, nY)), 0, 40).round()
    withdraw = np.clip(118 + rng.normal(0, 7, (1, nY)) + rng.normal(0, 3, (nT, nY)), 95, 145).round()
    p_wet = np.clip(reg.map(lambda r: r["wet"]).to_numpy()[:, None] + 0.25 * (year_anom[None, :]), .25, .85)
    # monsoon "break": a dry stretch in Jul-Aug, longer in bad years
    brk_start = rng.integers(35, 80, (nT, nY)) + rng.integers(-3, 4, (1, nY))
    brk_len = np.clip(rng.gamma(2.0, 4.0, (nT, nY)) - 25 * year_anom[None, :], 0, 30).round()

    day = np.arange(D)[None, None, :]
    in_monsoon = (day >= onset_true[..., None]) & (day <= withdraw[..., None])
    in_break = (day >= brk_start[..., None]) & (day < (brk_start + brk_len)[..., None])
    p = np.where(in_monsoon, p_wet[..., None], .08)
    p = np.where(in_break, .06, p)
    wet = np.zeros((nT, nY, D), bool)
    u = rng.random((nT, nY, D))
    for d in range(D):                                                    # Markov persistence
        prev = wet[..., d - 1] if d else np.zeros((nT, nY), bool)
        pd_ = np.where(prev, np.minimum(p[..., d] + .18, .95), np.maximum(p[..., d] - .12, .02))
        wet[..., d] = u[..., d] < pd_
    amt = np.where(wet, rng.gamma(.75, 1.0, (nT, nY, D)), 0.0)
    rain = amt / np.maximum(amt.sum(-1, keepdims=True), 1e-9) * season_total[..., None]

    seasonal = 1.8 * np.cos(np.linspace(0, np.pi, D)) - .3                # hotter June, cooler Aug-Sep
    tmean = (reg.map(lambda r: r["temp"]).to_numpy()[:, None, None] + talukas.temp_offset.to_numpy()[:, None, None]
             + rng.normal(0, .6, (1, nY, 1)) + seasonal[None, None, :] - 1.0 * wet + rng.normal(0, .8, (nT, nY, D)))
    tmax = tmean + 4.5 - 1.5 * wet + rng.normal(0, .5, (nT, nY, D))
    tmin = tmean - 4.5 + 1.0 * wet + rng.normal(0, .5, (nT, nY, D))
    return rain, tmax, tmin


def season_features(rain, tmax, tmin):
    """Exactly the definitions the Spark pipeline recomputes from the daily table."""
    nT, nY, D = rain.shape
    three = rain[..., :-2] + rain[..., 1:-1] + rain[..., 2:]
    hit = three >= 40
    onset = np.where(hit.any(-1), hit.argmax(-1), 45)                     # first 3-day total >= 40 mm
    dry = rain < 2.5
    longest = np.zeros((nT, nY), int); run = np.zeros((nT, nY), int)
    for d in range(SEP30 + 1):                                            # longest dry spell onset..30 Sep
        active = d >= onset
        run = np.where(active & dry[..., d], run + 1, 0)
        longest = np.maximum(longest, run)
    return dict(onset=onset, season_rain=rain.sum(-1), rainy_days=(rain >= 2.5).sum(-1),
                dry_spell=longest, mean_temp=((tmax + tmin) / 2).mean(-1))


# ------------------------------------------------------------------ main generator
def generate(n_farms: int, seed: int = SEED):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    RAW.mkdir(parents=True, exist_ok=True); TRUTH.mkdir(parents=True, exist_ok=True)
    (RAW / "weather_daily").mkdir(exist_ok=True); (RAW / "farm_seasons").mkdir(exist_ok=True)

    districts = pd.DataFrame(DISTRICTS, columns=["district_code", "district", "region", "normal_rain_mm",
                                                  "irrigated_share", "farm_weight"])
    districts.drop(columns="farm_weight").to_csv(RAW / "districts.csv", index=False)

    rows = []
    for dc, n in zip(districts.district_code, rng.integers(7, 15, len(districts))):
        for k in range(1, n + 1):
            rows.append((f"{dc}{k:02d}", dc, f"{dc}-WS{k:02d}"))
    talukas = pd.DataFrame(rows, columns=["taluka_id", "district_code", "station_id"])
    talukas["rain_factor"] = np.exp(rng.normal(0, .10, len(talukas)))
    talukas["temp_offset"] = rng.normal(0, .5, len(talukas))
    talukas["soil_ph_offset"] = rng.normal(0, .25, len(talukas))
    talukas[["taluka_id", "district_code", "station_id"]].to_csv(RAW / "talukas.csv", index=False)
    print(f"[{time.time()-t0:5.1f}s] {len(districts)} districts, {len(talukas)} talukas")

    # ---------------- weather
    rain, tmax, tmin = make_weather(rng, talukas, districts)
    feats = season_features(rain, tmax, tmin)
    dates = pd.date_range("2016-06-01", periods=SEASON_DAYS)
    for yi, y in enumerate(YEARS):
        ds = pd.date_range(f"{y}-06-01", periods=SEASON_DAYS).strftime("%Y-%m-%d")
        w = pd.DataFrame({
            "station_id": np.repeat(talukas.station_id.to_numpy(), SEASON_DAYS),
            "obs_date": np.tile(ds, len(talukas)),
            "rain_mm": rain[:, yi, :].ravel().round(1),
            "tmax_c": tmax[:, yi, :].ravel().round(1),
            "tmin_c": tmin[:, yi, :].ravel().round(1),
        })
        bad = rng.random(len(w)) < .001
        w.loc[bad, "rain_mm"] = -999.0                                   # sensor error sentinel
        w = w[rng.random(len(w)) > .004]                                 # station outages: missing days
        w.to_csv(RAW / "weather_daily" / f"weather_{y}.csv", index=False)
    print(f"[{time.time()-t0:5.1f}s] weather: {len(talukas) * SEASON_DAYS * len(YEARS):,} daily rows")

    # ---------------- farms
    w_t = districts.set_index("district_code").loc[talukas.district_code, "farm_weight"].to_numpy()
    farm_t = rng.choice(len(talukas), n_farms, p=w_t / w_t.sum())
    fd = districts.set_index("district_code").loc[talukas.district_code.to_numpy()[farm_t]].reset_index()
    freg = fd.region.map(REGIONS)
    size = np.clip(rng.lognormal(np.log(freg.map(lambda r: r["size"]).to_numpy()), .8), .1, 40).round(2)
    irrigated = (rng.random(n_farms) < fd.irrigated_share.to_numpy() * (1 + .3 * (size > 2))).astype(int)
    farms = pd.DataFrame({"farm_id": [f"F{i:06d}" for i in range(n_farms)],
                          "taluka_id": talukas.taluka_id.to_numpy()[farm_t], "area_ha": size,
                          "irrigated_2016": irrigated})
    farms.to_csv(RAW / "farms.csv", index=False)

    base_ph = fd.district_code.map(PH_OVERRIDE).fillna(freg.map(lambda r: r["ph"])).to_numpy()
    true_ph = np.clip(base_ph + talukas.soil_ph_offset.to_numpy()[farm_t] + rng.normal(0, .4, n_farms), 4.2, 9.2)
    true_oc = np.clip(rng.lognormal(np.log(freg.map(lambda r: r["oc"]).to_numpy()), .35), .1, 2.5)
    true_n = np.clip(rng.normal(220, 50, n_farms), 80, 450)               # available N kg/ha (SHC style)
    true_p = np.clip(rng.lognormal(np.log(18), .45, n_farms), 3, 90)
    true_k = np.clip(rng.lognormal(np.log(320), .35, n_farms), 80, 900)

    # ---------------- soil health cards (nested JSON, ~78 % of farms tested, some twice)
    tested = rng.random(n_farms) < .78
    second = tested & (rng.random(n_farms) < .35)
    with open(RAW / "soil_health_cards.jsonl", "w") as fh:
        card = 0
        for i in np.flatnonzero(tested | second):
            for cycle, start in ((1, "2015-07-01"), (2, "2017-07-01")) if second[i] else ((rng.integers(1, 3), None),):
                cyc = int(cycle)
                d0 = pd.Timestamp("2015-07-01" if cyc == 1 else "2017-07-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
                rec = {"card_id": f"SHC{card:07d}", "farm_id": farms.farm_id[i], "cycle": cyc,
                       "sample_date": d0.strftime("%Y-%m-%d"), "lab": f"STL-{farms.taluka_id[i][:3]}",
                       "results": {"pH": round(float(true_ph[i] + rng.normal(0, .15)), 1),
                                   "EC_dS_m": round(float(abs(rng.normal(.35, .15))), 2),
                                   "OC_pct": round(float(true_oc[i] * np.exp(rng.normal(0, .1))), 2),
                                   "N_kg_ha": round(float(true_n[i] * np.exp(rng.normal(0, .1)))),
                                   "P_kg_ha": round(float(true_p[i] * np.exp(rng.normal(0, .12))), 1),
                                   "K_kg_ha": round(float(true_k[i] * np.exp(rng.normal(0, .1))))}}
                if rng.random() < .01:
                    rec["results"]["pH"] = None                           # incomplete report
                fh.write(json.dumps(rec) + "\n"); card += 1
    print(f"[{time.time()-t0:5.1f}s] farms: {n_farms:,}; soil cards: {card:,}")

    # ---------------- farm seasons (panel: every farm, every year)
    d_code = fd.district_code.to_numpy()
    mix = np.array([MIX_OVERRIDE.get(dc, REGIONS[r]["mix"]) for dc, r in zip(d_code, fd.region)])
    mix = mix / mix.sum(1, keepdims=True)
    cum = mix.cumsum(1)
    crop = (rng.random((n_farms, 1)) > cum).sum(1)
    skill = np.exp(rng.normal(0, .08, n_farms))
    n_habit = np.exp(rng.normal(0, .25, n_farms))
    irr = irrigated.copy()
    outbreaks = {}                                                         # simulated pest outbreaks
    for y in LABEL_YEARS + [2026]:
        for c, k in ((2, 3), (1, 2)):                                      # cotton bollworm, soybean mosaic
            for dc in rng.choice(districts.district_code, k, replace=False):
                outbreaks[(y, dc, c)] = rng.uniform(.5, .7)
    pd.DataFrame([(y, dc, CROPS[c], round(1 - v, 2)) for (y, dc, c), v in outbreaks.items()],
                 columns=["season_year", "district_code", "crop", "loss_share"]).to_csv(TRUTH / "pest_outbreaks.csv", index=False)

    eval_rows, stream_frames = [], []
    total = 0
    for yi, y in enumerate(LABEL_YEARS + [2026]):
        if yi:                                                             # crop rotation + irrigation adoption
            switch = rng.random(n_farms) < .2
            crop = np.where(switch, (rng.random((n_farms, 1)) > cum).sum(1), crop)
            irr = np.where((irr == 0) & (rng.random(n_farms) < .015), 1, irr)
        prog = (y - 2016) / 10
        pr = np.array([.40 - .25 * prog, .50, .10 + .20 * prog])          # local / improved / hybrid
        var = (rng.random((n_farms, 1)) > pr.cumsum()[None, :]).sum(1).clip(0, 2)
        var = np.where(crop == 2, np.where(rng.random(n_farms) < .95, 2, 1), var)   # Bt cotton is hybrid
        cp = CROP_P.iloc[crop]
        n_rate = np.where(rng.random(n_farms) < np.where(crop == 2, .03, .08), 0,
                          cp.n_use.to_numpy() * n_habit * (1 + .015 * (y - 2016)) * np.exp(rng.normal(0, .15, n_farms)))
        n_rate = np.clip(n_rate, 0, 300)
        s_idx = (farm_t, yi)
        onset = feats["onset"][s_idx]
        delay = np.where(irr == 1, rng.normal(-2, 6, n_farms), rng.gamma(2.0, 4.0, n_farms))
        sow = pd.Timestamp(f"{y}-06-01") + pd.to_timedelta(np.clip(onset + delay.round(), 0, 75), unit="D")

        water = feats["season_rain"][s_idx] + 300 * irr
        f_w = _logistic(water, cp.w_half.to_numpy(), cp.w_slope.to_numpy()) * \
            (1 - cp.fl_max.to_numpy() * np.clip((water - cp.fl_start.to_numpy()) / 1000, 0, 1))
        dryk = cp.dry_k.to_numpy() * np.where(irr == 1, .25, 1)
        f_dry = np.clip(1 - dryk * np.maximum(0, feats["dry_spell"][s_idx] - 12), .4, 1)
        f_t = np.exp(-((feats["mean_temp"][s_idx] - cp.t_opt.to_numpy()) / cp.t_w.to_numpy()) ** 2)
        f_ph = np.exp(-((true_ph - cp.ph_opt.to_numpy()) / cp.ph_w.to_numpy()) ** 2)
        f_n = (1 - cp.n_zero.to_numpy() * np.exp(-(n_rate + 40 * true_oc) / cp.n_rate.to_numpy())) * \
            (1 - .0012 * np.clip(n_rate - 1.8 * cp.n_use.to_numpy(), 0, None))
        f_sow = np.clip(1 - .006 * np.maximum(0, delay - 10), .6, 1)
        f_var = np.array([VARIETY_MULT[v] for v in ("local", "improved", "hybrid")])[var]
        f_size = 1 - .03 * np.log(size / 1.4)
        signal = cp.ymax.to_numpy() * f_w * f_dry * f_t * f_ph * f_n * f_sow * f_var * f_size * skill
        yld = signal * np.exp(rng.normal(0, .11, n_farms))
        ob = np.array([outbreaks.get((y, dc, c), 1.0) for dc, c in zip(d_code, crop)])
        yld *= ob
        shock = rng.random(n_farms) < .02
        yld[shock] *= rng.uniform(.35, .7, shock.sum())

        rec = pd.DataFrame({
            "record_id": [f"R{y}{i:06d}" for i in range(n_farms)], "farm_id": farms.farm_id,
            "season_year": y, "crop": np.array(CROPS)[crop],
            "variety_type": np.array(["local", "improved", "hybrid"])[var],
            "sowing_date": sow.strftime("%Y-%m-%d"), "irrigated": irr,
            "nitrogen_kg_ha": n_rate.round(), "yield_value": yld.round(3), "yield_unit": "t/ha"})
        if y == 2026:                                                      # becomes the live stream
            rec = rec.drop(columns=["yield_value", "yield_unit"]).sample(min(30000, n_farms // 2), random_state=seed)
            rec.to_csv(RAW / "stream_source_2026.csv", index=False)
            pd.DataFrame({"record_id": rec.record_id, "true_yield_t_ha": yld[rec.index].round(3)}).to_csv(
                TRUTH / "yield_2026.csv", index=False)
            continue
        if y in (2024, 2025):
            eval_rows.append(pd.DataFrame({"crop": rec.crop, "y": yld, "s": signal}))   # outbreaks are not predictable
        # ---- data-quality problems typical of field surveys
        n = len(rec)
        m = rng.random(n)
        rec.loc[m < .15, "sowing_date"] = sow[m < .15].strftime("%d/%m/%Y")     # older app date format
        variants = {"paddy": ["Paddy", "PADDY", "rice", " paddy"], "soybean": ["Soyabean", "SOYBEAN", "soybean "],
                    "cotton": ["Cotton", "kapas"], "tur": ["Tur", "Arhar", "pigeonpea"]}
        mv = rng.random(n) < .06
        rec.loc[mv, "crop"] = [rng.choice(variants[c]) for c in rec.crop[mv]]
        q_dist = rng.choice(districts.district_code, 3, replace=False)          # some offices report q/ha
        q = np.isin(d_code, q_dist)
        rec.loc[q, "yield_value"] = (rec.loc[q, "yield_value"] * 10).round(2)
        rec.loc[q, "yield_unit"] = "q/ha"
        rec.loc[rng.random(n) < .03, "nitrogen_kg_ha"] = np.nan
        rec.loc[rng.random(n) < .01, "variety_type"] = None
        rec.loc[rng.random(n) < .0005, "nitrogen_kg_ha"] = 9999
        rec.loc[rng.random(n) < .0005, "yield_value"] = -1
        dup = rec.sample(frac=.003, random_state=seed + y)
        rec = pd.concat([rec, dup]).sample(frac=1, random_state=y)
        rec.to_csv(RAW / "farm_seasons" / f"survey_{y}.csv", index=False)
        total += len(rec)
        print(f"[{time.time()-t0:5.1f}s] survey_{y}.csv: {len(rec):,} rows")

    ev = pd.concat(eval_rows)
    ceiling = {c: round(float(1 - ((g.y - g.s) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()), 3)
               for c, g in ev.groupby("crop")}
    ceiling["all_crops"] = round(float(1 - ((ev.y - ev.s) ** 2).sum() / ((ev.y - ev.y.mean()) ** 2).sum()), 3)
    json.dump({"r2_ceiling_test_years": ceiling, "farm_season_rows": total}, open(TRUTH / "ceiling.json", "w"), indent=2)
    print(f"[{time.time()-t0:5.1f}s] done: {total:,} farm-season rows. R² ceiling (2024-25): {ceiling}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", type=int, default=200_000)
    generate(ap.parse_args().farms)
