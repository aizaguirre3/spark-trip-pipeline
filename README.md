# Distributed Trip-Fare Pipeline (PySpark)

An end-to-end **PySpark** pipeline that ingests partitioned trip data, performs
distributed cleaning and feature engineering, and trains a **Spark MLlib**
regression model to predict trip fares. Written entirely against the Spark
DataFrame + MLlib APIs, so the same code runs on a laptop (`local[*]`) or scales
unchanged across a multi-node cluster.

## Why this exists

A compact, reproducible demonstration of working with large datasets on a
distributed computing framework: partitioned Parquet I/O, shuffle-based window
aggregations, an ML `Pipeline` with indexing/one-hot encoding, and held-out
evaluation — plus tests and CI so it behaves like production code.

## Architecture

```
generate_data.py   ->  partitioned Parquet (by day_of_week)
                            |
pipeline.py         ->  read (distributed)
                        clean   (drop nulls / nonpositive fares / outliers, with counts)
                        engineer (time buckets, distance bins, per-hour demand window)
                        ML Pipeline (StringIndexer -> OneHotEncoder -> VectorAssembler -> LinearRegression)
                        evaluate (RMSE, R2 on an 80/20 split)
```

## Quickstart

```bash
pip install -r requirements.txt

# generate a synthetic dataset (scale --rows up on a cluster)
python src/generate_data.py --rows 200000 --out data/trips_raw

# run the distributed pipeline
python src/pipeline.py --data data/trips_raw
```

Example output:

```
=== Pipeline metrics ===
  total_rows: 200000
  dropped_null_distance: 3859
  dropped_nonpositive_fare: 1917
  dropped_distance_outliers: 218
  kept_rows: 194038
  test_rmse: 2.0169
  test_r2: 0.9898
```

## Scaling notes

- Data is **partitioned by `day_of_week`** on write, so reads prune partitions
  and parallelize across executors.
- `trips_in_hour` is computed with a **window aggregation** (`partitionBy`),
  exercising a real distributed shuffle.
- `spark.sql.shuffle.partitions` is set conservatively for local runs; raise it
  to match cluster cores for large data.
- To run against real data (e.g. NYC TLC trip records), point `--data` at the
  Parquet location and drop the generator step.

## Testing

```bash
pytest -v
```

Tests spin up a `local[2]` SparkSession and cover the generator, the cleaning
stage (verifying invalid rows are removed), feature engineering, model training,
and the full end-to-end run. Runs on Python 3.11 and 3.12 via GitHub Actions CI.

## Tech

Python · PySpark (DataFrame API, Spark SQL, MLlib) · Parquet · pytest · GitHub Actions
