"""Paths and the shared SparkSession builder (works on Windows, Linux and macOS)."""
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
LAKE = DATA / "lake"
BRONZE, SILVER, GOLD = LAKE / "bronze", LAKE / "silver", LAKE / "gold"
STREAM_IN = DATA / "stream" / "incoming"
STREAM_OUT = DATA / "stream" / "output"
CHECKPOINTS = DATA / "stream" / "checkpoints"
OUTPUTS = ROOT / "outputs"
ANALYTICS_OUT = OUTPUTS / "analytics"
MODEL_OUT = OUTPUTS / "model"
SERVING_OUT = OUTPUTS / "serving"

TRAIN_YEARS = range(2016, 2024)   # 2016-2023 train
TEST_YEARS = (2024, 2025)         # most recent seasons held out (temporal validation)
STREAM_YEAR = 2026                # "current" season arriving as a stream


def _java_roots() -> list[Path]:
    """Places Java installers actually use on Windows, including per-user installs."""
    env = os.environ
    return [Path(r) for r in (env.get("ProgramFiles", r"C:\Program Files"),
                              env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                              os.path.join(env.get("LOCALAPPDATA", ""), "Programs"),
                              env.get("ProgramData", r"C:\ProgramData"),
                              r"C:\Java") if r]


def _find_java() -> str | None:
    """Spark 3.5 needs Java 8/11/17. Installers often leave java.exe off PATH and may put it
    under %LOCALAPPDATA%, so search the usual places and prefer 17, then 11, then 8."""
    home = os.environ.get("JAVA_HOME")
    if home and (Path(home) / "bin" / "java.exe").exists():
        return home
    if shutil.which("java"):
        return ""                                        # already usable from PATH
    found = []
    for root in _java_roots():
        if not root.is_dir():
            continue
        for depth in ("*", "*/*", "*/*/*"):              # bounded search, no full disk walk
            for exe in root.glob(f"{depth}/bin/java.exe"):
                jdk = exe.parent.parent
                m = re.search(r"(?:jdk|jre)-?(?:1\.)?(8|11|17)\b", jdk.name, re.I)
                if not m:
                    continue
                rank = [17, 11, 8].index(int(m.group(1)))
                bundled = 1 if re.search(r"[\\/](Android|JetBrains|Unity|IntelliJ)[\\/]", str(jdk), re.I) else 0
                found.append((rank * 2 + bundled, str(jdk)))
    return min(found)[1] if found else None


def _prepare_windows() -> None:
    """Spark on Windows needs winutils.exe + hadoop.dll, a Java 8/11/17 runtime and an
    explicit Python path."""
    hadoop_home = ROOT / "hadoop"
    if (hadoop_home / "bin" / "winutils.exe").exists():
        os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
        os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    elif "HADOOP_HOME" not in os.environ:
        print("WARNING: hadoop\\bin\\winutils.exe not found. Run setup_windows.ps1 first, "
              "or Spark will fail when writing files.")

    java = _find_java()
    if java is None:
        print("WARNING: no Java 8/11/17 found. Run setup_windows.ps1, or install Temurin 17 "
              "from https://adoptium.net/temurin/releases/?version=17")
    elif java:                                      # found off-PATH -> make it usable
        os.environ["JAVA_HOME"] = java
        os.environ["PATH"] = str(Path(java) / "bin") + os.pathsep + os.environ.get("PATH", "")


def get_spark(app_name: str = "MaharashtraCropBDA", memory: str = "4g"):
    """Create (or reuse) a local SparkSession using all CPU cores."""
    os.environ["PYSPARK_PYTHON"] = sys.executable          # workers use this same Python
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if os.name == "nt":
        _prepare_windows()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "16")     # small cluster -> fewer shuffle partitions
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
        .config("spark.sql.adaptive.enabled", "true")     # adaptive query execution
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.warehouse.dir", str(DATA / "spark-warehouse"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def p(path: Path) -> str:
    """Spark-friendly path string (forward slashes work on every OS)."""
    return path.resolve().as_posix()
