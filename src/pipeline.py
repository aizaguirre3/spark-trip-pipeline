"""Distributed trip-fare pipeline on PySpark.

Stages:
  1. Read partitioned Parquet (distributed).
  2. Clean: drop nulls, filter invalid/outlier rows, surface dropped counts.
  3. Feature-engineer: time-of-day buckets, distance bins, per-hour demand
     via a window aggregation, one-hot/indexer pipeline.
  4. Train a Spark MLlib model (linear regression) inside an ML Pipeline.
  5. Evaluate on a held-out split (RMSE, R2) and report.

Everything uses the Spark DataFrame + MLlib APIs so it scales from a laptop
to a multi-node cluster unchanged.
"""
from __future__ import annotations

import argparse

from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def build_spark(app_name: str = "trip-fare-pipeline") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def clean(df: DataFrame) -> tuple[DataFrame, dict]:
    """Filter invalid rows; return cleaned frame and a dict of dropped counts."""
    total = df.count()
    null_dist = df.filter(F.col("trip_distance").isNull()).count()
    bad_fare = df.filter(F.col("fare_amount") <= 0).count()
    outliers = df.filter(F.col("trip_distance") > 200).count()

    cleaned = (
        df.filter(F.col("trip_distance").isNotNull())
        .filter(F.col("fare_amount") > 0)
        .filter(F.col("trip_distance") <= 200)
        .filter((F.col("passenger_count") >= 1) & (F.col("passenger_count") <= 6))
    )
    counts = {
        "total_rows": total,
        "dropped_null_distance": null_dist,
        "dropped_nonpositive_fare": bad_fare,
        "dropped_distance_outliers": outliers,
        "kept_rows": cleaned.count(),
    }
    return cleaned, counts


def engineer_features(df: DataFrame) -> DataFrame:
    """Derive model features, including a windowed per-hour demand signal."""
    df = df.withColumn(
        "time_bucket",
        F.when((F.col("pickup_hour") >= 6) & (F.col("pickup_hour") < 10), "morning_rush")
        .when((F.col("pickup_hour") >= 10) & (F.col("pickup_hour") < 16), "midday")
        .when((F.col("pickup_hour") >= 16) & (F.col("pickup_hour") < 20), "evening_rush")
        .otherwise("offpeak"),
    )
    df = df.withColumn(
        "distance_bin",
        F.when(F.col("trip_distance") < 2, "short")
        .when(F.col("trip_distance") < 8, "medium")
        .otherwise("long"),
    )
    # Window aggregation: trips-per-hour as a demand proxy (distributed shuffle)
    hour_window = Window.partitionBy("pickup_hour")
    df = df.withColumn("trips_in_hour", F.count("*").over(hour_window))
    return df


def build_pipeline() -> Pipeline:
    cat_cols = ["time_bucket", "distance_bin"]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_cols],
        outputCols=[f"{c}_ohe" for c in cat_cols],
    )
    assembler = VectorAssembler(
        inputCols=[
            "trip_distance",
            "passenger_count",
            "trips_in_hour",
            "time_bucket_ohe",
            "distance_bin_ohe",
        ],
        outputCol="features",
    )
    lr = LinearRegression(featuresCol="features", labelCol="fare_amount", maxIter=20)
    return Pipeline(stages=[*indexers, encoder, assembler, lr])


def run(spark: SparkSession, data_path: str) -> dict:
    raw = spark.read.parquet(data_path)
    cleaned, counts = clean(raw)
    feats = engineer_features(cleaned)

    train, test = feats.randomSplit([0.8, 0.2], seed=42)
    model = build_pipeline().fit(train)
    preds = model.transform(test)

    rmse = RegressionEvaluator(
        labelCol="fare_amount", predictionCol="prediction", metricName="rmse"
    ).evaluate(preds)
    r2 = RegressionEvaluator(
        labelCol="fare_amount", predictionCol="prediction", metricName="r2"
    ).evaluate(preds)

    metrics = {**counts, "test_rmse": round(rmse, 4), "test_r2": round(r2, 4)}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/trips_raw")
    args = parser.parse_args()

    spark = build_spark()
    metrics = run(spark, args.data)
    print("\n=== Pipeline metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    spark.stop()


if __name__ == "__main__":
    main()
