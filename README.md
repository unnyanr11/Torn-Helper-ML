# Torn-Helper ML

Backend-only prediction and model-research worker for Torn-Helper. It publishes
Python-generated forecasts to the assigned Supabase stock shards. There is no
website backtesting interface.

## Five-minute prediction workflow

`.github/workflows/prediction.yml` runs every five minutes (at minutes
03, 08, 13, 18, 23, 28, 33, 38, 43, 48, 53 and 58 UTC). It can also be
started manually with optional force-training and force-calibration switches.

Each bounded GitHub Actions run:

1. Connects to each configured shard and verifies its project assignment.
2. Reads new price observations, using a disposable cache containing public
   market prices only to avoid repeatedly downloading the full 180-day history.
3. Checks matured forecast outcomes against exact observed prices.
4. Retrains models when at least 10 minutes have elapsed since successful
   training; calibration is reconsidered every 3 hours.
5. Produces and stores the 104 configured stock/window/horizon forecasts.
6. Updates each shard's health heartbeat and processing metrics.

Training and calibration due times are stored in each shard's
`stock_engine.ml_v2_health` row, not inferred from the wall-clock schedule.
A run can be delayed by GitHub queueing. Runs do not overlap within this
repository, and per-project PostgreSQL advisory locks prevent simultaneous
workers from writing the same shard.

The worker processes up to four shard projects concurrently and has a 12-minute
job limit. It is a finite scheduled job, not an always-on server. Five-minute
scheduling is best-effort: GitHub may delay or drop scheduled runs.

## Required setup

1. Add the Actions repository secret `STOCK_ML_PROJECTS_JSON`. It must be a
   JSON object mapping each configured shard number (1-18) to that shard's
   PostgreSQL **Session pooler** connection URL. Use the same shard mapping
   already configured for the Torn-Helper Python prediction worker. Never
   commit URLs or credentials.
2. In each stock shard, install the live prediction schema by running
   `python/stock_ml/realtime_setup.sql`. This creates the private
   `stock_engine.ml_v2_*` tables and does not modify existing RLS policies.
   The worker expects the collector schema and helper function
   `stock_engine.collect_responses()` to already exist.
3. Ensure the connection role can read `stock_engine.settings`,
   `stock_engine.stocks`, and `stock_engine.prices`, call the collector
   helper, and read/write the `stock_engine.ml_v2_*` tables.
4. Enable GitHub Actions. Verify the first manual prediction run before
   relying on the five-minute schedule.

The prediction workflow uses the same `STOCK_ML_PROJECTS_JSON` secret as the
manual backtest workflow. No secret is embedded in source or printed in logs.

## Database ownership

Each stock is written only to its assigned shard:

| Project | Stock IDs |
|---|---|
| 1-17 | 2 per project: `project * 2 - 1` and `project * 2` |
| 18 | 35 |

The main application database remains separate from the stock-data shards.
The worker validates project assignments before writing. Its database schema
is isolated in `stock_engine`; this repository does not change RLS policies.

## Model and safety behavior

- The live model registry contains 104 combinations: 12 short windows x 6
  short horizons, plus 8 long windows x 4 long horizons.
- The adaptive linear model selects Ridge, ElasticNet or a Ridge/ElasticNet
  blend using chronological validation with horizon purging.
- Forecasts that fail validation, lack sufficient history, have stale prices,
  or exceed supported prediction bounds are not forced into Buy/Sell signals.
- Calibration uses matured, non-overlapping forward outcomes. Insufficient
  samples or no demonstrated holdout improvement leave calibration inactive.
- Forecasts are saved in `stock_engine.ml_v2_latest`; forward checks and
  health data are saved separately.

The prediction worker is not a guarantee of profitability. Forecasts are
estimates, not financial advice, and actual execution prices, spreads and
liquidity may differ from stored closes.

## Backtesting (automatic every three hours)

`.github/workflows/backtest.yml` runs the complete backtest matrix every
three hours (UTC), and can also be started manually. A unit-test job runs once
before the stock matrix begins. The matrix contains one independent job for
each of the 35 stocks, with up to eight stock jobs running concurrently and
fail-fast disabled so one stock's failure does not cancel the others.

Each stock tests all 104 live registry window/horizon combinations (12 short
windows x 6 short horizons, plus 8 long windows x 4 long horizons) across all
six supported backtest model families. The scheduled defaults use 180 days of
history, a 30-day out-of-sample evaluation period and up to 120 walk-forward
folds per configuration. Long horizons may still be skipped by the evaluator
when the available history cannot support a valid matured sample.

Backtest predictions, trades, metrics and calibration candidates are stored
in the separate `stock_engine.ml_backtest_*` tables. Ridge-compatible
candidates can be promoted to live calibration only after at least 30 matured
live-forward checks and at least a 2% MAE improvement against the current
calibration. Other model families are comparison-only and are not wired into
the live predictor. No RLS policies are changed.

Each stock job has a 165-minute timeout. The workflow allows up to eight
concurrent stock jobs to limit database connection pressure. If a run takes
longer than the three-hour schedule interval, GitHub's workflow-level
concurrency setting queues the next run rather than running two full batches
at once.

Before running backtests, install `python/stock_ml/backtest_setup.sql` in each
shard after the live prediction schema is present. This creates isolated
backtest tables and does not alter RLS policies.

## Tests

```sh
python -m pip install -r python/stock_ml/requirements.txt
python -m unittest discover -s python/stock_ml -v
```

No credentials, private holdings, account data or personal information belong
in this public repository.
