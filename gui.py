import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from crop_yield_core import (
    FEATURE_LABELS,
    FEATURE_SPECS,
    FEATURES,
    RESULTS_DIR,
    TARGET,
    load_data,
    load_saved_bundle,
    predict_yield,
    save_artifacts,
    train_bundle,
)

INK = "#13261c"
MUTED = "#3f5548"
LEAF = "#1f6b45"
MOSS = "#2f8f5b"
SIDEBAR = "#153325"
SIDEBAR_TEXT = "#f4faf5"
MAIN_BG = "#f4f7f2"
CARD = "#ffffff"
HERO = "#1d4a34"
DEFAULTS = {f: float(spec[4]) for f, spec in FEATURE_SPECS.items()}


def fmt_value(feature: str, value: float) -> str:
    step = FEATURE_SPECS[feature][3]
    if feature == "irrigated":
        return "yes" if value >= 0.5 else "no"
    decimals = 0 if step >= 1 else len(str(step).split(".")[1])
    return f"{value:.{decimals}f}"


def resolve_ui_font(root: tk.Misc | None = None) -> str:
    temp = None
    if root is None:
        temp = tk.Tk()
        temp.withdraw()
        root = temp
    try:
        families = {name.lower(): name for name in tkfont.families(root)}
        for candidate in ("Aptos", "Times New Roman", "Times"):
            if candidate.lower() in families:
                return families[candidate.lower()]
        return "Times New Roman"
    finally:
        if temp is not None:
            temp.destroy()


class FieldYieldApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FieldYield | Smart Agriculture")
        self.geometry("1100x800")
        self.minsize(960, 640)
        self.configure(bg=MAIN_BG)

        self.ui_font = resolve_ui_font(self)
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [self.ui_font, "Times New Roman", "Times"],
            }
        )

        self.bundle = None
        self.page = tk.StringVar(value="Predict")
        self.sliders = {}
        self.result_vars = {
            "yield": tk.StringVar(value="—"),
            "model": tk.StringVar(value="—"),
            "mae": tk.StringVar(value="—"),
        }
        self.status_var = tk.StringVar(value="Loading model…")

        self._build_layout()
        self.after(80, self._bootstrap_model)

    def _font(self, size: int = 11, weight: str = "normal") -> tuple:
        return (self.ui_font, size, weight)

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        nav = tk.Frame(self, bg=SIDEBAR, width=220)
        nav.grid(row=0, column=0, sticky="nsw")
        nav.grid_propagate(False)

        tk.Label(
            nav,
            text="FieldYield",
            bg=SIDEBAR,
            fg=SIDEBAR_TEXT,
            font=self._font(18, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(22, 4))
        tk.Label(
            nav,
            text="Smart Agriculture",
            bg=SIDEBAR,
            fg="#c7e0d0",
            font=self._font(10),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 18))

        for name in ("Predict", "Insights", "About"):
            btn = tk.Button(
                nav,
                text=name,
                font=self._font(12),
                bg=SIDEBAR,
                fg=SIDEBAR_TEXT,
                activebackground=LEAF,
                activeforeground=SIDEBAR_TEXT,
                relief="flat",
                anchor="w",
                padx=18,
                pady=10,
                cursor="hand2",
                command=lambda n=name: self._show_page(n),
            )
            btn.pack(fill="x", padx=8, pady=2)

        tk.Button(
            nav,
            text="Retrain models",
            font=self._font(11, "bold"),
            bg=SIDEBAR_TEXT,
            fg=SIDEBAR,
            activebackground="#ffffff",
            activeforeground=INK,
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
            command=self._retrain,
        ).pack(side="bottom", fill="x", padx=16, pady=16)

        right = tk.Frame(self, bg=MAIN_BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        hero = tk.Frame(right, bg=HERO)
        hero.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        tk.Label(
            hero,
            text="FieldYield",
            bg=HERO,
            fg=SIDEBAR_TEXT,
            font=self._font(22, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 2))
        tk.Label(
            hero,
            text="Paddy yield (t/ha) from weather, soil-health-card values, nitrogen, irrigation and farm size.",
            bg=HERO,
            fg="#e7f4ec",
            font=self._font(11),
            anchor="w",
            wraplength=760,
            justify="left",
        ).pack(fill="x", padx=16, pady=(0, 14))

        self.content = tk.Frame(right, bg=MAIN_BG)
        self.content.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 8))
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        self.pages = {
            "Predict": self._build_predict_page(self.content),
            "Insights": self._build_insights_page(self.content),
            "About": self._build_about_page(self.content),
        }

        status = tk.Label(
            right,
            textvariable=self.status_var,
            bg=MAIN_BG,
            fg=MUTED,
            font=self._font(9),
            anchor="w",
        )
        status.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))

        self._show_page("Predict")

    def _build_predict_page(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent, bg=MAIN_BG)
        frame.columnconfigure(0, weight=1)

        tk.Label(
            frame,
            text="Estimate yield for a farm",
            bg=MAIN_BG,
            fg=INK,
            font=self._font(16, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 10))

        card = tk.Frame(frame, bg=CARD, highlightbackground="#c9d9cf", highlightthickness=1)
        card.grid(row=1, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        defaults = DEFAULTS

        for i, feature in enumerate(FEATURES):
            col = i % 2
            row = i // 2
            box = tk.Frame(card, bg=CARD)
            box.grid(row=row, column=col, sticky="ew", padx=14, pady=12)
            box.columnconfigure(0, weight=1)

            value_var = tk.DoubleVar(value=defaults[feature])
            label_var = tk.StringVar(value=f"{FEATURE_LABELS[feature]}: {fmt_value(feature, defaults[feature])}")

            tk.Label(box, textvariable=label_var, bg=CARD, fg=INK, font=self._font(11), anchor="w").grid(
                row=0, column=0, sticky="ew"
            )
            scale = tk.Scale(
                box,
                from_=0,
                to=100,
                orient="horizontal",
                resolution=FEATURE_SPECS[feature][3],
                showvalue=False,
                bg=CARD,
                fg=INK,
                troughcolor="#d7e6db",
                highlightthickness=0,
                font=self._font(9),
                command=lambda _v, f=feature: self._on_slider(f),
            )
            scale.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self.sliders[feature] = {
                "scale": scale,
                "value": value_var,
                "label": label_var,
                "lo": 0.0,
                "hi": 100.0,
            }

        tk.Button(
            frame,
            text="Predict yield",
            font=self._font(12, "bold"),
            bg=LEAF,
            fg="#ffffff",
            activebackground=MOSS,
            activeforeground="#ffffff",
            relief="flat",
            padx=18,
            pady=8,
            cursor="hand2",
            command=self._predict,
        ).grid(row=2, column=0, sticky="w", pady=(14, 10))

        metrics = tk.Frame(frame, bg=MAIN_BG)
        metrics.grid(row=3, column=0, sticky="ew")
        for i, (title, key) in enumerate(
            (("Predicted yield (t/ha)", "yield"), ("Model", "model"), ("Typical abs. error (MAE, t/ha)", "mae"))
        ):
            cell = tk.Frame(metrics, bg=CARD, highlightbackground="#c9d9cf", highlightthickness=1)
            cell.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            metrics.columnconfigure(i, weight=1)
            tk.Label(cell, text=title, bg=CARD, fg=MUTED, font=self._font(9), anchor="w").pack(
                fill="x", padx=12, pady=(10, 0)
            )
            tk.Label(
                cell,
                textvariable=self.result_vars[key],
                bg=CARD,
                fg=INK,
                font=self._font(16, "bold"),
                anchor="w",
            ).pack(fill="x", padx=12, pady=(2, 10))

        tk.Label(
            frame,
            text="Result is paddy yield in tonnes per hectare, learned from a simulated dataset — an educational estimate, not an agronomic recommendation.",
            bg=MAIN_BG,
            fg=MUTED,
            font=self._font(9),
            anchor="w",
            wraplength=820,
            justify="left",
        ).grid(row=4, column=0, sticky="ew", pady=(10, 0))
        return frame

    def _build_insights_page(self, parent: tk.Frame) -> tk.Frame:
        outer = tk.Frame(parent, bg=MAIN_BG)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=MAIN_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        frame = tk.Frame(canvas, bg=MAIN_BG)
        self._insights_window = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(self._insights_window, width=e.width),
        )

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        self._insights_canvas = canvas

        tk.Label(
            frame,
            text="Model performance & data signals",
            bg=MAIN_BG,
            fg=INK,
            font=self._font(16, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        table_wrap = tk.Frame(frame, bg=CARD, highlightbackground="#c9d9cf", highlightthickness=1)
        table_wrap.pack(fill="x", pady=(0, 12))

        columns = ("model", "cv_mae", "cv_mean", "mae", "rmse", "r2")
        style = ttk.Style()
        style.configure(
            "Insights.Treeview",
            font=self._font(10),
            rowheight=28,
            background=CARD,
            fieldbackground=CARD,
            foreground=INK,
        )
        style.configure(
            "Insights.Treeview.Heading",
            font=self._font(10, "bold"),
            background="#e7efe4",
            foreground=INK,
        )
        style.map("Insights.Treeview", background=[("selected", "#cfe3d5")])

        self.metrics_table = ttk.Treeview(
            table_wrap,
            columns=columns,
            show="headings",
            height=6,
            style="Insights.Treeview",
        )
        headings = {
            "model": ("Model", 190),
            "cv_mae": ("CV MAE", 90),
            "cv_mean": ("CV R²", 90),
            "mae": ("Test MAE", 90),
            "rmse": ("Test RMSE", 90),
            "r2": ("Test R²", 90),
        }
        for key, (title, width) in headings.items():
            self.metrics_table.heading(key, text=title, anchor="center")
            self.metrics_table.column(key, width=width, anchor="center", stretch=True)
        self.metrics_table.column("model", anchor="w")

        yscroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.metrics_table.yview)
        self.metrics_table.configure(yscrollcommand=yscroll.set)
        self.metrics_table.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=8)
        yscroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

        self.chart_host = tk.Frame(frame, bg=MAIN_BG)
        self.chart_host.pack(fill="both", expand=True)
        self._chart_canvas = None
        return outer

    def _build_about_page(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent, bg=MAIN_BG)
        tk.Label(
            frame,
            text="About this project",
            bg=MAIN_BG,
            fg=INK,
            font=self._font(16, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        self.about_text = tk.Label(
            frame,
            text="",
            bg=MAIN_BG,
            fg=INK,
            font=self._font(11),
            justify="left",
            anchor="nw",
            wraplength=820,
        )
        self.about_text.pack(fill="both", expand=True, anchor="nw")
        return frame

    def _show_page(self, name: str) -> None:
        self.page.set(name)
        for page_name, frame in self.pages.items():
            frame.grid_forget()
        self.pages[name].grid(row=0, column=0, sticky="nsew")
        if name == "Insights" and self.bundle is not None:
            self._render_insights()

    def _bootstrap_model(self) -> None:
        try:
            saved = load_saved_bundle()
            if saved and saved.get("bounds") is not None and saved.get("metrics_df") is not None:
                self.bundle = saved
                self.status_var.set("Loaded saved model from crop_yield_results/")
            else:
                self.status_var.set("Training models…")
                self.update_idletasks()
                bundle = train_bundle(load_data())
                save_artifacts(bundle)
                self.bundle = bundle
                self.status_var.set("Models trained and saved.")
            self._apply_bounds()
            self._update_about()
            if self.page.get() == "Insights":
                self._render_insights()
        except Exception as exc:
            self.status_var.set("Failed to load model.")
            messagebox.showerror("FieldYield", str(exc))

    def _retrain(self) -> None:
        try:
            self.status_var.set("Retraining models…")
            self.update_idletasks()
            bundle = train_bundle(load_data())
            save_artifacts(bundle)
            self.bundle = bundle
            self._apply_bounds()
            self._update_about()
            if self.page.get() == "Insights":
                self._render_insights()
            self.status_var.set("Models retrained.")
            messagebox.showinfo("FieldYield", "Models retrained successfully.")
        except Exception as exc:
            self.status_var.set("Retrain failed.")
            messagebox.showerror("FieldYield", str(exc))

    def _apply_bounds(self) -> None:
        bounds = self.bundle["bounds"]
        defaults = DEFAULTS
        for feature, widgets in self.sliders.items():
            lo = float(bounds.loc["min", feature])
            hi = float(bounds.loc["max", feature])
            value = float(np.clip(defaults[feature], lo, hi))
            widgets["lo"] = lo
            widgets["hi"] = hi
            widgets["scale"].configure(from_=lo, to=hi)
            widgets["scale"].set(value)
            widgets["value"].set(value)
            widgets["label"].set(f"{FEATURE_LABELS[feature]}: {fmt_value(feature, value)}")

    def _on_slider(self, feature: str) -> None:
        widgets = self.sliders[feature]
        value = float(widgets["scale"].get())
        widgets["value"].set(value)
        widgets["label"].set(f"{FEATURE_LABELS[feature]}: {fmt_value(feature, value)}")

    def _predict(self) -> None:
        if self.bundle is None:
            messagebox.showwarning("FieldYield", "Model is still loading.")
            return
        values = {feature: float(widgets["value"].get()) for feature, widgets in self.sliders.items()}
        try:
            estimate = predict_yield(self.bundle["best_model"], self.bundle["bounds"], values)
            mae = float(self.bundle["metrics_df"].loc[self.bundle["best_name"], "MAE"])
            self.result_vars["yield"].set(f"{estimate:.2f}")
            self.result_vars["model"].set(str(self.bundle["best_name"]))
            self.result_vars["mae"].set(f"{mae:.2f}")
            self.status_var.set("Prediction ready.")
        except ValueError as exc:
            messagebox.showerror("FieldYield", str(exc))

    def _update_about(self) -> None:
        model_name = self.bundle["best_name"] if self.bundle else "—"
        self.about_text.configure(
            text=(
                "FieldYield predicts paddy yield (t/ha) from eight farm inputs using supervised regression.\n\n"
                "Dataset: ~3,000 simulated paddy farms (data/crop_yield_data.csv), generated by\n"
                "generate_data.py from agronomic response curves with noise, missing values and entry errors.\n"
                f"Selected model: {model_name} (chosen by 5-fold cross-validated MAE)\n"
                "Libraries: Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Tkinter\n"
                "Pipeline: clean → explore → impute → train/compare models → evaluate → predict\n\n"
                "Yield responds non-linearly (optimum rainfall, temperature and pH; diminishing returns to nitrogen),\n"
                "so tree ensembles beat linear models. About 20-30 % of yield variation is unexplainable noise.\n"
                "This is an educational demo, not a tool for real farming decisions."
            )
        )

    def _render_insights(self) -> None:
        if self.bundle is None:
            return

        metrics = self.bundle["metrics_df"].round(3)
        for item in self.metrics_table.get_children():
            self.metrics_table.delete(item)
        for model_name, row in metrics.iterrows():
            self.metrics_table.insert(
                "",
                "end",
                values=(
                    model_name,
                    f"{row['CV_MAE']:.3f}",
                    f"{row['CV_R2_mean']:.3f}",
                    f"{row['MAE']:.3f}",
                    f"{row['RMSE']:.3f}",
                    f"{row['R2']:.3f}",
                ),
            )

        if self._chart_canvas is not None:
            self._chart_canvas.get_tk_widget().destroy()
            self._chart_canvas = None

        df = self.bundle["df"] if "df" in self.bundle else load_data()
        pred_path = RESULTS_DIR / "test_predictions.csv"
        if pred_path.exists():
            preds = pd.read_csv(pred_path)
        else:
            preds = None

        # a standalone Figure (not plt.subplots) so pyplot does not create/close its own window
        width_px = max(self._insights_canvas.winfo_width() - 20, 800)
        fig = Figure(figsize=(width_px / 100, 11.5), dpi=100)
        grid = fig.subplots(3, 2)
        # keep the original 2x3 indexing used below: axes[r, c] -> 3x2 grid position
        axes = np.array([[grid[0, 0], grid[0, 1], grid[1, 0]], [grid[1, 1], grid[2, 0], grid[2, 1]]])
        fig.patch.set_facecolor(MAIN_BG)
        for ax in axes.ravel():
            ax.set_facecolor(CARD)

        short = {c: c.split("_")[0] for c in FEATURES + [TARGET]}
        sns.heatmap(
            df.rename(columns=short).corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cmap="RdYlGn",
            center=0,
            ax=axes[0, 0],
            cbar_kws={"shrink": 0.8},
        )
        axes[0, 0].set_title("Feature correlations")

        for flag, colour, label in ((0, "#c8682b", "rain-fed"), (1, LEAF, "irrigated")):
            part = df[df["irrigated"] == flag]
            axes[0, 1].scatter(part["rainfall_mm"], part[TARGET], alpha=0.25, s=8, color=colour, label=label)
        axes[0, 1].set_xlabel("Rainfall (mm)")
        axes[0, 1].set_ylabel("Yield (t/ha)")
        axes[0, 1].set_title("Yield vs rainfall")
        axes[0, 1].legend(fontsize=8, frameon=False)

        axes[0, 2].scatter(df["soil_ph"], df[TARGET], alpha=0.2, s=8, color=MOSS)
        axes[0, 2].set_xlabel("Soil pH")
        axes[0, 2].set_ylabel("Yield (t/ha)")
        axes[0, 2].set_title("Yield vs soil pH (optimum ≈ 6.5)")

        if self.bundle.get("importance") is not None:
            imp = pd.Series(self.bundle["importance"]).sort_values(ascending=True)
            imp.plot(kind="barh", ax=axes[1, 0], color=MOSS)
            axes[1, 0].set_title("Permutation importance")
            axes[1, 0].set_xlabel("Increase in MAE (t/ha) when shuffled")
        else:
            axes[1, 0].text(0.5, 0.5, "Importance unavailable", ha="center", va="center")
            axes[1, 0].set_axis_off()

        mae_series = metrics["MAE"].sort_values(ascending=True)
        axes[1, 1].barh(mae_series.index.astype(str), mae_series.values, color=LEAF)
        axes[1, 1].set_title("Model MAE (lower is better)")
        axes[1, 1].set_xlabel("Test MAE (t/ha)")

        if preds is not None:
            axes[1, 2].scatter(
                preds["actual_yield"],
                preds["predicted_yield"],
                s=12,
                alpha=0.45,
                color=LEAF,
            )
            lo = min(preds["actual_yield"].min(), preds["predicted_yield"].min())
            hi = max(preds["actual_yield"].max(), preds["predicted_yield"].max())
            axes[1, 2].plot([lo, hi], [lo, hi], "r--", linewidth=1)
            axes[1, 2].set_xlabel("Actual yield (t/ha)")
            axes[1, 2].set_ylabel("Predicted yield (t/ha)")
            axes[1, 2].set_title("Held-out predictions")
        else:
            axes[1, 2].text(0.5, 0.5, "Predictions file missing", ha="center", va="center")
            axes[1, 2].set_axis_off()

        for ax in fig.axes:
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(10)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        # Inside a scrollable Canvas the first layout pass gives the chart ~1 px of height and
        # matplotlib (on <Configure> and <Map>) would shrink the figure to match. Keep its own size.
        widget.unbind("<Configure>")
        widget.unbind("<Map>")
        widget.pack(anchor="nw")
        self._chart_canvas = canvas
        self._insights_canvas.configure(scrollregion=self._insights_canvas.bbox("all"))


def main() -> None:
    app = FieldYieldApp()
    app.mainloop()


if __name__ == "__main__":
    main()
