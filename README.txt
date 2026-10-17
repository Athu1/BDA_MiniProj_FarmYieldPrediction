FieldYield - Big Data Analytics on Maharashtra kharif crop yields (simulated data)

WINDOWS (one time):   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
then every time:      .\.venv\Scripts\activate

RUN EVERYTHING:       python run_all.py            (~15 min on a 4-core laptop; add --fast for a quicker model)
DASHBOARD:            python dashboard\app.py
LIVE STREAM DEMO:     terminal 1:  python -m bda.streaming --reset
                      terminal 2:  python stream_simulator.py --reset
SPARK DEMO:           python spark_demo.py   (then open http://localhost:4040)
NOTEBOOK:             jupyter notebook BDA_Crop_Yield_Analysis.ipynb

Project layout
  spark_demo.py          guided live demo of Spark + Web UI for the viva
  data_generator.py      simulates the raw landing zone (2 M farm-season records + weather + soil cards)
  stream_simulator.py    replays 2026 submissions as a live JSON stream
  bda/config.py          paths + SparkSession (handles Windows winutils automatically)
  bda/cleaning.py        schemas and cleaning rules shared by batch and streaming
  bda/ingest.py          raw -> Bronze -> Silver Parquet, data-quality report
  bda/features.py        Silver -> Gold: weather season aggregates, 5-table join
  bda/analytics.py       12 Spark SQL analyses -> outputs/analytics/*.csv
  bda/train_model.py     Spark MLlib: linear baseline + tuned gradient-boosted trees
  bda/streaming.py       Structured Streaming: live scoring + windowed counts
  bda/benchmark.py       pandas vs Spark, CSV vs Parquet, partition pruning
  bda/serving_model.py   compact scikit-learn model used by the dashboard
  dashboard/app.py       Tkinter dashboard (reads outputs/, no Spark needed)
  outputs/               results shipped with the project so the dashboard works immediately

The full explanation is in the accompanying PDF.
