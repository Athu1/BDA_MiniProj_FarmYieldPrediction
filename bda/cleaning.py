"""Schemas and cleaning rules shared by batch ingestion and the streaming job."""
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (DoubleType, IntegerType, StringType, StructField, StructType)

FARM_SEASON_SCHEMA = StructType([
    StructField("record_id", StringType()), StructField("farm_id", StringType()),
    StructField("season_year", IntegerType()), StructField("crop", StringType()),
    StructField("variety_type", StringType()), StructField("sowing_date", StringType()),
    StructField("irrigated", IntegerType()), StructField("nitrogen_kg_ha", DoubleType()),
    StructField("yield_value", DoubleType()), StructField("yield_unit", StringType()),
])
STREAM_SCHEMA = StructType([f for f in FARM_SEASON_SCHEMA.fields if f.name not in ("yield_value", "yield_unit")]
                           + [StructField("event_time", StringType())])
WEATHER_SCHEMA = StructType([
    StructField("station_id", StringType()), StructField("obs_date", StringType()),
    StructField("rain_mm", DoubleType()), StructField("tmax_c", DoubleType()), StructField("tmin_c", DoubleType()),
])
FARMS_SCHEMA = StructType([
    StructField("farm_id", StringType()), StructField("taluka_id", StringType()),
    StructField("area_ha", DoubleType()), StructField("irrigated_2016", IntegerType()),
])

CROP_ALIASES = {"paddy": "paddy", "rice": "paddy", "soybean": "soybean", "soyabean": "soybean",
                "cotton": "cotton", "kapas": "cotton", "tur": "tur", "arhar": "tur", "pigeonpea": "tur"}
MAX_YIELD_T_HA = {"paddy": 12.0, "soybean": 5.0, "cotton": 6.0, "tur": 4.0}   # physical plausibility


def normalise_crop(col: Column) -> Column:
    key = F.lower(F.trim(col))
    mapping = F.create_map(*[F.lit(x) for kv in CROP_ALIASES.items() for x in kv])
    return mapping[key]


def parse_mixed_date(col: Column) -> Column:
    """Two app versions wrote dates differently: ISO yyyy-MM-dd and dd/MM/yyyy."""
    return F.coalesce(F.to_date(col, "yyyy-MM-dd"), F.to_date(col, "dd/MM/yyyy"))


def clean_farm_seasons(df: DataFrame, with_yield: bool = True) -> DataFrame:
    out = (df
           .withColumn("crop", normalise_crop("crop"))
           .withColumn("sowing_date", parse_mixed_date(F.col("sowing_date")))
           .withColumn("variety_type", F.coalesce(F.lower(F.trim("variety_type")), F.lit("unknown")))
           .withColumn("nitrogen_kg_ha", F.when(F.col("nitrogen_kg_ha").between(0, 400), F.col("nitrogen_kg_ha"))))
    if with_yield:
        max_y = F.create_map(*[F.lit(x) for kv in MAX_YIELD_T_HA.items() for x in kv])
        out = (out.withColumn("yield_t_ha", F.when(F.col("yield_unit") == "q/ha", F.col("yield_value") / 10)
                              .otherwise(F.col("yield_value")))
               .filter((F.col("yield_t_ha") > 0) & (F.col("yield_t_ha") <= max_y[F.col("crop")]))
               .drop("yield_value", "yield_unit"))
    return out.filter(F.col("crop").isNotNull() & F.col("sowing_date").isNotNull())
