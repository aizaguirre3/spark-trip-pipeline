# Claude Code Handoff — Push spark-trip-pipeline to GitHub

Hand this whole file to Claude Code (e.g. `claude` in the project folder, then paste
the "PROMPT FOR CLAUDE CODE" section below).

---

## What this project is

A working **PySpark + Spark MLlib** distributed data pipeline. It generates a
partitioned trip dataset, cleans it, engineers features (including a windowed
per-hour demand aggregation), and trains a Spark MLlib linear-regression model
with held-out RMSE/R² evaluation. It is meant for a Data Scientist resume to
demonstrate distributed-computing / large-dataset experience.

**It already runs.** Verified locally: 6/6 unit tests pass, and a 200K-row run
produced R² ≈ 0.99, RMSE ≈ $2.02.

## File structure

```
spark-trip-pipeline/
├── README.md                     # project overview, quickstart, scaling notes
├── requirements.txt              # pyspark, pytest
├── .gitignore                    # ignores generated data/, spark cruft, __pycache__
├── .github/workflows/ci.yml      # GitHub Actions CI (Python 3.11 & 3.12, JDK 17)
├── src/
│   ├── generate_data.py          # synthetic partitioned-Parquet generator
│   └── pipeline.py               # clean → feature-engineer → MLlib train → evaluate
└── tests/
    └── test_pipeline.py          # 6 tests on a local[2] SparkSession
```

## Before pushing — verify it still runs (optional but recommended)

```bash
pip install -r requirements.txt
export SPARK_LOCAL_IP=127.0.0.1     # avoids a hostname-resolution warning
pytest -v                            # expect 6 passed
python src/generate_data.py --rows 200000 --out data/trips_raw
python src/pipeline.py --data data/trips_raw   # prints metrics
```

`data/` is gitignored, so generated Parquet will NOT be committed — good.

---

## PROMPT FOR CLAUDE CODE

> I have a project folder `spark-trip-pipeline` that I want pushed to my GitHub.
> Please:
> 1. Verify the tests pass: install requirements and run `pytest -v` (set
>    `SPARK_LOCAL_IP=127.0.0.1` first). Report the result.
> 2. Initialize git, stage everything, and make an initial commit with the
>    message: "Distributed trip-fare pipeline (PySpark + Spark MLlib)".
> 3. Create a NEW public GitHub repo named `spark-trip-pipeline` under my account
>    and push to `main`. Use the `gh` CLI if it's authenticated
>    (`gh repo create spark-trip-pipeline --public --source=. --push`); otherwise
>    tell me what you need from me to authenticate.
> 4. Confirm the repo URL and tell me whether the GitHub Actions CI run started.
> 5. Add a CI status badge to the very top of README.md pointing at the new repo's
>    Actions workflow, then commit and push that change.
>
> Do NOT commit the `data/` directory (it's gitignored). Report the final repo URL
> when done.

---

## After it's pushed

Send me (back in the chat) the repo URL — e.g.
`https://github.com/<you>/spark-trip-pipeline` — and I'll add it as a clickable
"GitHub" link on the Spark project line in your resume and rebuild the .docx.

## Notes / gotchas

- **Authentication is yours to do.** Claude Code can run `git` and `gh`, but it
  can't create your GitHub account or log in as you — if `gh` isn't already
  authenticated, it'll prompt you (`gh auth login`).
- The CI workflow installs JDK 17 + PySpark and runs the 6 tests on every push.
  Check the **Actions** tab for a green check; that badge is a nice trust signal
  for anyone reviewing the repo.
- If you'd rather keep the repo private, swap `--public` for `--private` (but a
  reviewer following your resume link needs it public to see it).
