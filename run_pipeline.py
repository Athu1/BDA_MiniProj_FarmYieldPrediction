from crop_yield_core import TARGET, load_data, predict_yield, save_artifacts, train_bundle


def main() -> None:
    df = load_data()
    report = df.attrs["cleaning_report"]
    print(f"Read {report['rows_read']:,} rows | duplicates removed: {report['duplicates_removed']} | "
          f"implausible rows removed: {report['implausible_removed']} | kept: {report['rows_kept']:,}")
    print(f"Missing values left for imputation: {report['missing_by_feature']}")
    print(df.describe().round(2).T)

    bundle = train_bundle(df)
    print("\nModel comparison (selected by 5-fold CV MAE on training data; test set used once):")
    print(bundle["metrics_df"].round(3))
    print(f"\nSelected model: {bundle['best_name']}")
    print("\nPermutation importance (increase in test MAE, t/ha, when a feature is shuffled):")
    print(bundle["importance"].round(3))

    save_artifacts(bundle)

    demo = {
        "rainfall_mm": 1200, "avg_temp_c": 27.0, "sunlight_hours": 7.0, "soil_ph": 6.8,
        "organic_carbon_pct": 0.55, "nitrogen_kg_ha": 110, "irrigated": 1, "farm_size_ha": 1.5,
    }
    est = predict_yield(bundle["best_model"], bundle["bounds"], demo)
    print(f"\nDemo farm prediction ({bundle['best_name']}): {est:.2f} {TARGET.split('_', 1)[1].replace('_', '/')}")
    print("Artifacts saved to crop_yield_results/")


if __name__ == "__main__":
    main()
