"""Unit tests for the Spark trip-fare pipeline.

Uses a local[2] SparkSession so tests run in CI without a cluster.
"""
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import generate  # noqa: E402
from pipeline import clean, engineer_features, build_pipeline, run  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder.master("local[2]")
        .appName("tests")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    yield s
    s.stop()


@pytest.fixture(scope="session")
def raw(spark):
    return generate(spark, rows=5000)


def test_generator_row_count(raw):
    assert raw.count() == 5000


def test_generator_has_expected_columns(raw):
    expected = {
        "trip_id", "passenger_count", "trip_distance", "pickup_hour",
        "day_of_week", "rate_code", "fare_amount",
    }
    assert expected.issubset(set(raw.columns))


def test_clean_removes_invalid_rows(raw):
    cleaned, counts = clean(raw)
    # No nulls, no nonpositive fares, no extreme outliers remain
    assert cleaned.filter("trip_distance IS NULL").count() == 0
    assert cleaned.filter("fare_amount <= 0").count() == 0
    assert cleaned.filter("trip_distance > 200").count() == 0
    assert counts["kept_rows"] <= counts["total_rows"]
    assert counts["kept_rows"] > 0


def test_feature_engineering_adds_columns(raw):
    cleaned, _ = clean(raw)
    feats = engineer_features(cleaned)
    for col in ("time_bucket", "distance_bin", "trips_in_hour"):
        assert col in feats.columns
    # time_bucket only takes the four defined values
    buckets = {r.time_bucket for r in feats.select("time_bucket").distinct().collect()}
    assert buckets.issubset({"morning_rush", "midday", "evening_rush", "offpeak"})


def test_pipeline_trains_and_scores(spark, raw):
    cleaned, _ = clean(raw)
    feats = engineer_features(cleaned)
    train, test = feats.randomSplit([0.8, 0.2], seed=1)
    model = build_pipeline().fit(train)
    preds = model.transform(test)
    assert "prediction" in preds.columns
    assert preds.count() > 0


def test_end_to_end_run_returns_metrics(spark, raw, tmp_path):
    path = str(tmp_path / "trips")
    raw.write.mode("overwrite").parquet(path)
    metrics = run(spark, path)
    assert "test_rmse" in metrics and metrics["test_rmse"] >= 0
    assert "test_r2" in metrics
    assert metrics["kept_rows"] > 0
