"""
FieldYield BDA Dashboard - desktop front-end for the Spark pipeline outputs.

Reads only the small result files in outputs/ (no Spark needed), so it opens instantly.
Run after `python run_all.py`:   python dashboard/app.py
"""
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
AN = OUT / "analytics"
INK, MUTED, LEAF, MOSS, SIDEBAR, BG, CARD, ACCENT = ("#13261c", "#4a5f52", "#1f6b45", "#2f8f5b", "#153325",
                                                    "#f4f7f2", "#ffffff", "#c8682b")
CROP_COL = {"paddy": "#1f6b45", "soybean": "#c8a227", "cotton": "#2a6f8f", "tur": "#c8682b"}
PAGES = ["Overview", "District explorer", "Weather impact", "Anomalies", "Predict", "Live stream", "Model & benchmark"]


def font(size=10, weight="normal"):
    fam = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"
    return (fam, size, weight)


def read_csv(name):
    path = AN / name if (AN / name).exists() else OUT / name
    return pd.read_csv(path) if path.exists() else None


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FieldYield · Maharashtra Crop Big Data Analytics")
        self.geometry("1320x840"); self.minsize(1180, 760); self.configure(bg=BG)
        if not (AN / "kpis.csv").exists():
            messagebox.showerror("FieldYield", "No pipeline outputs found.\nRun:  python run_all.py   first.")
        self.data = {n: read_csv(f"{n}.csv") for n in (
            "kpis", "state_crop_trend", "district_crop_year", "rainfall_impact", "dry_spell_impact",
            "sowing_delay_impact", "yield_anomalies", "district_ranking", "nitrogen_response", "soil_ph_response",
            "variety_adoption", "benchmark")}
        self.quality = json.load(open(OUT / "quality_report.json")) if (OUT / "quality_report.json").exists() else {}
        self.metrics = json.load(open(OUT / "model" / "metrics.json")) if (OUT / "model" / "metrics.json").exists() else {}
        sp = OUT / "serving" / "serving_model.joblib"
        self.serving = joblib.load(sp) if sp.exists() else None
        ceil = ROOT / "data" / "truth" / "ceiling.json"
        self.ceiling = json.load(open(ceil))["r2_ceiling_test_years"] if ceil.exists() else {}

        style = ttk.Style(self); style.theme_use("clam")
        style.configure("Treeview", font=font(9), rowheight=22); style.configure("Treeview.Heading", font=font(9, "bold"))
        style.configure("TCombobox", padding=4)
        self.columnconfigure(1, weight=1); self.rowconfigure(0, weight=1)
        nav = tk.Frame(self, bg=SIDEBAR, width=210); nav.grid(row=0, column=0, sticky="ns"); nav.grid_propagate(False)
        tk.Label(nav, text="FieldYield", bg=SIDEBAR, fg="white", font=font(17, "bold"), anchor="w").pack(fill="x", padx=16, pady=(20, 0))
        tk.Label(nav, text="Maharashtra crop BDA", bg=SIDEBAR, fg="#bcd6c5", font=font(9), anchor="w").pack(fill="x", padx=16, pady=(0, 16))
        self.body = tk.Frame(self, bg=BG); self.body.grid(row=0, column=1, sticky="nsew", padx=16, pady=14)
        self.buttons = {}
        for name in PAGES:
            b = tk.Button(nav, text=name, font=font(11), bg=SIDEBAR, fg="white", activebackground=LEAF, relief="flat",
                          anchor="w", padx=16, pady=8, cursor="hand2", command=lambda n=name: self.show(n))
            b.pack(fill="x", padx=6, pady=1); self.buttons[name] = b
        tk.Label(nav, text="Simulated data · educational project", bg=SIDEBAR, fg="#8fb19c", font=font(8),
                 wraplength=180, justify="left").pack(side="bottom", padx=14, pady=14, anchor="w")
        self.show("Overview")

    # ------------------------------------------------------------ helpers
    def clear(self, title, subtitle=""):
        for w in self.body.winfo_children():
            w.destroy()
        for n, b in self.buttons.items():
            b.configure(bg=LEAF if n == self.current else SIDEBAR)
        tk.Label(self.body, text=title, bg=BG, fg=INK, font=font(17, "bold"), anchor="w").pack(fill="x")
        if subtitle:
            tk.Label(self.body, text=subtitle, bg=BG, fg=MUTED, font=font(10), anchor="w", justify="left",
                     wraplength=1050).pack(fill="x", pady=(0, 8))

    def figure(self, parent, w=10, h=4.2):
        fig = Figure(figsize=(w, h), dpi=100, facecolor=BG)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    def table(self, parent, df, height=8, widths=None):
        tv = ttk.Treeview(parent, columns=list(df.columns), show="headings", height=height)
        for i, c in enumerate(df.columns):
            tv.heading(c, text=c); tv.column(c, width=(widths or {}).get(c, 110), anchor="w" if i == 0 else "center")
        for row in df.itertuples(index=False):
            tv.insert("", "end", values=list(row))
        tv.pack(fill="x", pady=4)
        return tv

    def cards(self, parent, items):
        row = tk.Frame(parent, bg=BG); row.pack(fill="x", pady=(0, 10))
        for i, (label, value) in enumerate(items):
            c = tk.Frame(row, bg=CARD, highlightbackground="#cfdcd3", highlightthickness=1)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0)); row.columnconfigure(i, weight=1)
            tk.Label(c, text=label, bg=CARD, fg=MUTED, font=font(9), anchor="w").pack(fill="x", padx=10, pady=(8, 0))
            tk.Label(c, text=value, bg=CARD, fg=INK, font=font(16, "bold"), anchor="w").pack(fill="x", padx=10, pady=(0, 8))

    def show(self, name):
        self.current = name
        getattr(self, "page_" + name.split()[0].lower().replace("&", ""))()

    # ------------------------------------------------------------ pages
    def page_overview(self):
        self.clear("Overview", "Ten kharif seasons (2016-2025) of farm-level records processed with Apache Spark: "
                               "raw CSV/JSON → Bronze → Silver → Gold Parquet lake → SQL analytics, MLlib model and streaming.")
        k = self.data["kpis"]
        if k is not None:
            k = k.iloc[0]
            self.cards(self.body, [("Farm-season records", f"{int(k.farm_seasons):,}"), ("Farms", f"{int(k.farms):,}"),
                                   ("Districts / talukas", f"{int(k.districts)} / {int(k.talukas)}"),
                                   ("Seasons", f"{int(k.first_year)}–{int(k.last_year)}"), ("Irrigated", f"{k.irrigated_pct}%")])
        mid = tk.Frame(self.body, bg=BG); mid.pack(fill="both", expand=True)
        left = tk.Frame(mid, bg=BG); left.pack(side="left", fill="both", expand=True)
        fig, _ = self.figure(left, 7.5, 4.6)
        t = self.data["state_crop_trend"]
        axes = fig.subplots(2, 2, sharex=True)
        for ax, (crop, g) in zip(axes.ravel(), t.groupby("crop")):
            ax.plot(g.season_year, g.mean_yield, marker="o", color=CROP_COL[crop]); ax.set_title(f"{crop} — mean t/ha", fontsize=9)
            ax.tick_params(labelsize=7); ax.set_facecolor(CARD)
        fig.tight_layout()
        right = tk.Frame(mid, bg=BG, width=310); right.pack(side="left", fill="y", padx=(10, 0))
        tk.Label(right, text="Data-quality report (ingestion)", bg=BG, fg=INK, font=font(11, "bold"), anchor="w").pack(fill="x")
        q = pd.DataFrame([(k.replace("_", " "), f"{v:,}" if isinstance(v, int) else v) for k, v in self.quality.items()],
                         columns=["check", "count"])
        self.table(right, q, height=18, widths={"check": 215, "count": 85})

    def page_district(self):
        self.clear("District explorer", "Pick a district and crop: ten-season yield and rainfall history from the Spark SQL cube.")
        d = self.data["district_crop_year"]
        bar = tk.Frame(self.body, bg=BG); bar.pack(fill="x")
        dv, cv = tk.StringVar(value="Yavatmal"), tk.StringVar(value="cotton")
        ttk.Combobox(bar, textvariable=dv, values=sorted(d.district.unique()), state="readonly", width=24).pack(side="left")
        ttk.Combobox(bar, textvariable=cv, values=sorted(d.crop.unique()), state="readonly", width=12).pack(side="left", padx=8)
        holder = tk.Frame(self.body, bg=BG); holder.pack(fill="both", expand=True, pady=8)

        def draw(*_):
            for w in holder.winfo_children():
                w.destroy()
            g = d[(d.district == dv.get()) & (d.crop == cv.get())].sort_values("season_year")
            st = self.data["state_crop_trend"]; st = st[st.crop == cv.get()]
            if g.empty:
                tk.Label(holder, text="This crop is (almost) not grown in this district.", bg=BG, fg=MUTED, font=font(11)).pack()
                return
            fig, _ = self.figure(holder, 10, 5)
            a1, a2 = fig.subplots(2, 1, sharex=True)
            a1.plot(g.season_year, g.mean_yield, marker="o", color=CROP_COL[cv.get()], label=dv.get())
            a1.plot(st.season_year, st.mean_yield, ls="--", color="grey", label="Maharashtra")
            a1.set_ylabel("mean yield (t/ha)"); a1.legend(frameon=False, fontsize=8)
            a1.set_title(f"{cv.get()} in {dv.get()} · {int(g.farm_records.sum()):,} farm-seasons", fontsize=10)
            cols = [ACCENT if v < -19 else ("#2a6f8f" if v > 19 else "#9aa5a0") for v in g.rain_departure_pct]
            a2.bar(g.season_year, g.rain_departure_pct, color=cols); a2.axhline(0, color="k", lw=.6)
            a2.set_ylabel("rain vs normal (%)"); a2.set_xlabel("season")
            for ax in (a1, a2):
                ax.set_facecolor(CARD); ax.tick_params(labelsize=8)
            fig.tight_layout()
        dv.trace_add("write", draw); cv.trace_add("write", draw); draw()

    def page_weather(self):
        self.clear("Weather impact", "How rainfall departure, monsoon dry spells and late sowing change yields — "
                                     "and how much irrigation protects against them.")
        fig, _ = self.figure(self.body, 10.5, 6.2)
        a = fig.subplots(2, 2)
        ri, ds, sd = self.data["rainfall_impact"], self.data["dry_spell_impact"], self.data["sowing_delay_impact"]
        order = ["deficient", "normal", "excess"]; x = np.arange(3); w = .2
        for i, crop in enumerate(CROP_COL):
            g = ri[(ri.crop == crop) & (ri.irrigated == 0)].set_index("rain_category").reindex(order)
            a[0, 0].bar(x + (i - 1.5) * w, g.mean_yield, w, color=CROP_COL[crop], label=crop)
        a[0, 0].set_xticks(x, order); a[0, 0].set_title("Rain-fed farms: yield by monsoon category", fontsize=9)
        a[0, 0].legend(fontsize=7, frameon=False)
        for crop in CROP_COL:
            for irr, ls in ((0, "-"), (1, "--")):
                g = ds[(ds.crop == crop) & (ds.irrigated == irr)]
                a[0, 1].plot(g.dry_spell_bucket, g.mean_yield, ls, marker="o", ms=3, color=CROP_COL[crop])
        a[0, 1].set_title("Longest dry spell (solid rain-fed, dashed irrigated)", fontsize=9)
        for crop in CROP_COL:
            g = sd[(sd.crop == crop) & (sd.irrigated == 0)]
            a[1, 0].plot(g.delay_bucket_days, g.mean_yield, marker="o", ms=3, color=CROP_COL[crop])
        a[1, 0].set_title("Sowing delay after monsoon onset (rain-fed)", fontsize=9); a[1, 0].set_xlabel("days")
        nr = self.data["nitrogen_response"]
        for crop in CROP_COL:
            g = nr[nr.crop == crop]; a[1, 1].plot(g.n_bucket_kg_ha, g.mean_yield, marker="o", ms=3, color=CROP_COL[crop])
        a[1, 1].set_title("Nitrogen applied (kg/ha) vs yield", fontsize=9)
        for ax in a.ravel():
            ax.set_facecolor(CARD); ax.tick_params(labelsize=7); ax.set_ylabel("t/ha", fontsize=8)
        fig.tight_layout()

    def page_anomalies(self):
        self.clear("Anomalies", "District-crop-seasons whose mean yield fell more than 1.8 standard deviations below their "
                                "own ten-year history. Normal rainfall + collapse ⇒ pest or disease suspected.")
        an = self.data["yield_anomalies"]
        cols = ["district", "crop", "season_year", "mean_yield", "hist_mean", "z_score", "rain_departure_pct", "likely_cause"]
        self.table(self.body, an[cols], height=12, widths={"district": 160, "likely_cause": 300})
        fig, _ = self.figure(self.body, 10, 3)
        ax = fig.subplots()
        cnt = an.groupby(["season_year", "likely_cause"]).size().unstack(fill_value=0)
        cnt.plot(kind="bar", stacked=True, ax=ax, color=[ACCENT, "#2a6f8f", "#6b4fa0"][:cnt.shape[1]], width=.7)
        ax.set_title("Anomalies per season by likely cause", fontsize=9); ax.legend(fontsize=7, frameon=False)
        ax.set_facecolor(CARD); ax.tick_params(labelsize=8, rotation=0); fig.tight_layout()

    def page_predict(self):
        self.clear("Predict a farm's yield", "Uses the compact scikit-learn serving model (same features as the Spark MLlib model). "
                                            "Choosing a district pre-fills its typical weather and soil.")
        if self.serving is None:
            tk.Label(self.body, text="Serving model missing — run python run_all.py", bg=BG, fg=ACCENT).pack(); return
        S = self.serving; dd = S["district_defaults"].set_index("district")
        form = tk.Frame(self.body, bg=CARD, highlightbackground="#cfdcd3", highlightthickness=1); form.pack(fill="x")
        v = {}
        def combo(r, c, label, key, values, init):
            tk.Label(form, text=label, bg=CARD, fg=INK, font=font(10)).grid(row=r, column=c, sticky="w", padx=10, pady=5)
            v[key] = tk.StringVar(value=init)
            cb = ttk.Combobox(form, textvariable=v[key], values=values, state="readonly", width=20)
            cb.grid(row=r, column=c + 1, sticky="w", padx=6); return cb
        def num(r, c, label, key, init):
            tk.Label(form, text=label, bg=CARD, fg=INK, font=font(10)).grid(row=r, column=c, sticky="w", padx=10, pady=5)
            v[key] = tk.StringVar(value=str(init)); tk.Entry(form, textvariable=v[key], width=10, font=font(10)).grid(row=r, column=c + 1, sticky="w", padx=6)
        dist_cb = combo(0, 0, "District", "district", sorted(dd.index), "Yavatmal")
        combo(0, 2, "Crop", "crop", S["categories"]["crop"], "cotton")
        combo(0, 4, "Variety", "variety_type", ["local", "improved", "hybrid"], "hybrid")
        combo(1, 0, "Irrigated", "irrigated", ["0", "1"], "0")
        num(1, 2, "Nitrogen (kg/ha)", "nitrogen_kg_ha", 110); num(1, 4, "Farm area (ha)", "area_ha", 1.5)
        num(2, 0, "Sowing delay (days)", "sowing_delay_days", 5); num(2, 2, "Season rain (mm)", "season_rain_mm", 0)
        num(2, 4, "Longest dry spell (days)", "longest_dry_spell_days", 0); num(3, 0, "Rainy days", "rainy_days", 0)
        num(3, 2, "Mean temp (°C)", "mean_temp_c", 0); num(3, 4, "Soil pH", "soil_ph", 0)
        num(4, 0, "Organic carbon (%)", "soil_oc_pct", 0); num(4, 2, "Soil N (kg/ha)", "soil_n_kg_ha", 0)
        num(4, 4, "Soil P (kg/ha)", "soil_p_kg_ha", 0); num(5, 0, "Soil K (kg/ha)", "soil_k_kg_ha", 0)

        def prefill(*_):
            r = dd.loc[v["district"].get()]
            for k in ("season_rain_mm", "longest_dry_spell_days", "rainy_days", "mean_temp_c", "soil_ph", "soil_oc_pct",
                      "soil_n_kg_ha", "soil_p_kg_ha", "soil_k_kg_ha"):
                v[k].set(f"{r[k]:.2f}".rstrip("0").rstrip("."))
        v["district"].trace_add("write", prefill); prefill()
        result = tk.StringVar(value="—"); context = tk.StringVar(value="")

        def predict():
            try:
                r = dd.loc[v["district"].get()]
                row = {k: float(v[k].get()) for k in S["features"] if k in v and k not in ("crop", "variety_type")}
                row.update(crop=v["crop"].get(), variety_type=v["variety_type"].get(), region=r.region,
                           district_code=r.district_code)
                for k, (lo, hi) in S["ranges"].items():
                    if not lo <= row[k] <= hi:
                        raise ValueError(f"{k} = {row[k]} is outside the range seen in the data ({lo:.1f} – {hi:.1f}).")
                X = pd.DataFrame([row])[S["features"]]
                for c, cats in S["categories"].items():
                    X[c] = pd.Categorical(X[c], categories=cats)
                y = float(S["model"].predict(X)[0])
                m = S["metrics"].get(row["crop"], {})
                hist = self.data["district_crop_year"]
                h = hist[(hist.district == v["district"].get()) & (hist.crop == row["crop"])].mean_yield.mean()
                result.set(f"{y:.2f} t/ha")
                context.set(f"Typical error for {row['crop']}: ±{m.get('mae', float('nan')):.2f} t/ha (MAE).   "
                            f"10-season district average: {h:.2f} t/ha" if not np.isnan(h) else "")
            except ValueError as e:
                messagebox.showerror("FieldYield", str(e))
        tk.Button(self.body, text="Predict yield", font=font(11, "bold"), bg=LEAF, fg="white", relief="flat", padx=18, pady=7,
                  command=predict).pack(anchor="w", pady=10)
        tk.Label(self.body, textvariable=result, bg=BG, fg=INK, font=font(26, "bold")).pack(anchor="w")
        tk.Label(self.body, textvariable=context, bg=BG, fg=MUTED, font=font(10)).pack(anchor="w")

    def page_live(self):
        self.clear("Live stream", "2026 season submissions scored by Spark Structured Streaming. Start "
                                  "`python stream_simulator.py` and `python -m bda.streaming` in two terminals, then watch this page.")
        status = tk.StringVar(); holder = tk.Frame(self.body, bg=BG)
        tk.Label(self.body, textvariable=status, bg=BG, fg=INK, font=font(11, "bold"), anchor="w").pack(fill="x")
        holder.pack(fill="both", expand=True)
        self._auto = getattr(self, "_auto", tk.BooleanVar(value=True))
        tk.Checkbutton(self.body, text="auto-refresh every 5 s", variable=self._auto, bg=BG).pack(anchor="w")

        def refresh():
            if self.current != "Live stream":
                return
            for w in holder.winfo_children():
                w.destroy()
            sj, sc = OUT / "stream_status.json", OUT / "stream_summary.csv"
            if not sj.exists():
                status.set("No stream output yet."); return
            s = json.load(open(sj))
            status.set(f"{s['records']:,} records scored · last micro-batch {s['last_batch_rows']} rows in "
                       f"{s['last_batch_seconds']} s · updated {s['updated']}")
            df = pd.read_csv(sc)
            fig, _ = self.figure(holder, 10, 3.4)
            a1, a2 = fig.subplots(1, 2)
            by = df.groupby("crop").apply(lambda g: np.average(g.mean_predicted_yield, weights=g.records), include_groups=False)
            a1.bar(by.index, by.values, color=[CROP_COL[c] for c in by.index]); a1.set_title("Predicted 2026 mean yield (t/ha)", fontsize=9)
            top = df.groupby("district").records.sum().nlargest(12)
            a2.barh(top.index[::-1], top.values[::-1], color=MOSS); a2.set_title("Records received by district (top 12)", fontsize=9)
            for ax in (a1, a2):
                ax.set_facecolor(CARD); ax.tick_params(labelsize=8)
            fig.tight_layout()
            show = df.sort_values("records", ascending=False).head(10)
            self.table(holder, show, height=6, widths={"district": 180})
            if self._auto.get():
                self.after(5000, refresh)
        refresh()

    def page_model(self):
        self.clear("Model & benchmark", "Spark MLlib results on the held-out 2024–2025 seasons (trained on 2016–2023), "
                                       "versus the best score any model could reach given the noise in the data.")
        m = self.metrics
        if m:
            rows = []
            for crop in ["overall", "paddy", "soybean", "cotton", "tur"]:
                rows.append((crop, m["linear_regression"][crop]["r2"], m["gbt"][crop]["r2"], m["gbt"][crop]["mae"],
                             self.ceiling.get(crop if crop != "overall" else "all_crops", "")))
            self.table(self.body, pd.DataFrame(rows, columns=["crop", "linear R²", "GBT R²", "GBT MAE (t/ha)", "R² ceiling"]), height=5)
            fig, _ = self.figure(self.body, 10, 2.5)
            ax = fig.subplots()
            imp = pd.Series(m["feature_importance"]).sort_values()
            ax.barh(imp.index, imp.values, color=MOSS); ax.set_title("GBT feature importance", fontsize=9)
            ax.set_facecolor(CARD); ax.tick_params(labelsize=7); fig.tight_layout()
        b = self.data["benchmark"]
        if b is not None:
            tk.Label(self.body, text="Benchmark (seconds unless stated)", bg=BG, fg=INK, font=font(11, "bold"), anchor="w").pack(fill="x", pady=(8, 0))
            self.table(self.body, b, height=6, widths={"test": 330, "pandas_or_left": 95, "spark_or_right": 95, "note": 380})


if __name__ == "__main__":
    Dashboard().mainloop()
