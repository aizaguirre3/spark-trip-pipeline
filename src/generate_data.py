"""Generate a large, realistic synthetic trip dataset (NYC-taxi style).

Writes partitioned Parquet so the pipeline exercises real distributed-read paths.
Designed to scale: pass --rows 50_000_000 on a cluster, or a small value locally.
"""
from __future__ import annotations

import argparse

from pyspark.sql import SparkSession, functions as F


def build_spark(app_name: str = "trip-data-generator") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def generate(spark: SparkSession, rows: int):
    """Build a synthetic trip dataset with intentional 'dirt' to clean downstream."""
    df = spark.range(rows).withColumnRenamed("id", "trip_id")

    # Pseudo-random but deterministic features derived from the id + rand()
    df = (
        df.withColumn("passenger_count", (F.rand(seed=1) * 6 + 1).cast("int"))
        .withColumn("trip_distance", F.round(F.rand(seed=2) * 25 + 0.1, 2))
        .withColumn("pickup_hour", (F.rand(seed=3) * 24).cast("int"))
        .withColumn("day_of_week", (F.rand(seed=4) * 7 + 1).cast("int"))
        .withColumn("rate_code", (F.rand(seed=5) * 5 + 1).cast("int"))
    )

    # Fare loosely a function of distance + passengers + time-of-day surge, plus noise
    df = df.withColumn(
        "fare_amount",
        F.round(
            2.5
            + F.col("trip_distance") * 2.75
            + F.col("passenger_count") * 0.4
            + F.when(
                (F.col("pickup_hour") >= 17) & (F.col("pickup_hour") <= 20), 4.0
            ).otherwise(0.0)
            + (F.rand(seed=6) - 0.5) * 6,
            2,
        ),
    )

    # Inject realistic dirt: ~2% null distances, ~1% negative fares, a few absurd outliers.
    # Compute the dirtied columns in a single select to avoid self-referential
    # withColumn lineage that the Spark analyzer rejects.
    df = df.select(
        "trip_id",
        "passenger_count",
        "pickup_hour",
        "day_of_week",
        "rate_code",
        F.when(F.rand(seed=7) < 0.02, F.lit(None))
        .when(F.rand(seed=9) < 0.001, F.lit(9999.0))
        .otherwise(F.col("trip_distance"))
        .alias("trip_distance"),
        F.when(F.rand(seed=8) < 0.01, F.col("fare_amount") * -1)
        .otherwise(F.col("fare_amount"))
        .alias("fare_amount"),
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--out", default="data/trips_raw")
    args = parser.parse_args()

    spark = build_spark()
    df = generate(spark, args.rows)
    # Partition by day_of_week to demonstrate partitioned distributed writes/reads
    df.write.mode("overwrite").partitionBy("day_of_week").parquet(args.out)
    print(f"Wrote {args.rows:,} rows to {args.out} (partitioned by day_of_week)")
    spark.stop()


if __name__ == "__main__":
    main()
