# Torn-Helper ML

Short-term, backend-only stock prediction and model-research worker for [Torn-Helper](https://github.com/unnyanr11/Torn-Helper). This repository owns minute-based inference, training, calibration, matured-outcome checks and short-term walk-forward backtesting. It publishes results to assigned private Supabase stock shards; it does not provide a website or long-term weekly forecasts.

**Long-term forecasts live in [LongTerm-Torn-Helper-ML](https://github.com/unnyanr11/LongTerm-Torn-Helper-ML).** Do not add weekly/long-horizon model logic or write long-term records into this repository's stock shards.


## Project scope and architecture

- **This repository:** short-term minute data, short-term model training/inference, calibration, forward outcome checks and short-term backtests.
- **Main app:** [Torn-Helper](https://github.com/unnyanr11/Torn-Helper) displays forecasts and charts; it is not the training worker.
- **Long-term worker:** [LongTerm-Torn-Helper-ML](https://github.com/unnyanr11/LongTerm-Torn-Helper-ML) owns weekly aggregation, long-horizon models and long-term backtests.

The live registry contains 12 analysis windows (15 minutes, 30 minutes, 1, 2, 3, 6, 12, 18, 24, 72, 168 and 720 hours) and six horizons (1, 3, 6, 12, 18 and 24 hours): 72 combinations per stock. All 35 stocks are distributed over 18 configured stock-data projects. Projects 1–17 own two stock IDs each; project 18 owns stock 35.

The feature-based linear competition compares Ridge, ElasticNet and a Ridge/ElasticNet blend over chronological training/selection/holdout periods. It purges overlapping future labels for the selected horizon, evaluates matured outcomes and retains a no-change baseline. Model selection and calibration are research controls, not a guarantee that a forecast will be profitable.

## Five-minute prediction workflow

`.github/workflows/prediction.yml` is started manually and then re-dispatches
itself after a 300-second timer. This creates an approximately five-minute
best-effort cycle, not a cron schedule or an exact service-level interval.
GitHub queueing, job duration and dispatch latency can shift start times.
The workflow can also be started manually with optional force-training and
force-calibration switches; automatic re-dispatch explicitly requests
force-training.

Each bounded GitHub Actions run:

1. Connects to each configured shard and verifies its project assignment.
2. Reads new price observations, using a disposable cache containing public
   market prices only to avoid repeatedly downloading the full 180-day history.
3. Checks matured forecast outcomes against exact observed prices.
4. Trains short-term models when due (or when force-training is requested);
   calibration is reconsidered every 3 hours unless explicitly forced.
5. Produces and stores the 72 configured short-term stock/window/horizon forecasts.
6. Updates each shard's health heartbeat and processing metrics.

Training and calibration due times are stored in each shard's
`stock_engine.ml_v2_health` row, not inferred from the wall-clock schedule.
A run can be delayed by GitHub queueing. Runs do not overlap within this
repository, and per-project PostgreSQL advisory locks prevent simultaneous
workers from writing the same shard.

The worker processes up to four shard projects concurrently and has a 12-minute
job limit. It is a finite scheduled job, not an always-on server. Five-minute
scheduling is best-effort: GitHub may delay or drop scheduled runs.

## Aggressive historical minute-price backfill

Run `.github/workflows/historical-backfill.yml` manually from the Actions tab
to fill missing minute observations directly in SG1-SG18. The default is a full 365-day historical request. This archive is independent
of the short-term model worker, which may continue to use a shorter training
retention window. Tornsy may not have a complete year available for every
stock; fetched counts and actual earliest/latest timestamps must be audited
before treating the archive as a complete year. SG1-SG17 each process their two
assigned stocks; SG18 processes stock 35. The worker validates each database's
`stock_engine.settings` project number and exact stock assignment before
writing.

The backfill reads the public Tornsy historical endpoint at
`https://tornsy.com/api/{ticker}?interval=m1&from=...&to=...&limit=2000`.
It requests 1,800-minute non-overlapping windows, retries transient provider
errors with backoff, and rate-limits requests globally across shards. It inserts
only real, minute-aligned observations and uses conflict-safe inserts, so
existing live observations are not overwritten. Missing source minutes remain
missing; the worker never interpolates or fabricates prices.

All writes go to each stock's assigned `stock_engine.prices` table. No price
history is written to the main Torn-Helper database, and the backfill does not
alter RLS policies. A dry-run option fetches and validates the provider data
without writing. The workflow requires the existing
`STOCK_ML_PROJECTS_JSON` Actions secret to contain all 18 shard connection
URLs. Inspect the per-stock fetched/inserted counts and the final shard summary
in the workflow logs. Tornsy's documented endpoint supports minute history,
time-bounded requests and a maximum page limit of 2,000 rows; provider
availability and actual historical coverage determine how many rows can be
recovered.

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

## Trading thresholds and costs

The short-term live signal and backtest default minimum expected net return is **0.1%** after estimated trading costs. The default live round-trip cost estimate is 0.20%: the game's 0.10% fee is charged once when selling, plus the model's assumed 0.05% slippage on each side. The backtest deducts the fee once and slippage on both entry and exit. There is no maximum expected-profit threshold. Configure the worker/backtest defaults with GitHub Actions repository Variables:

- `SHORT_MIN_EXPECTED_NET_RETURN` (fraction; default `0.001` = 0.1%)
- `SHORT_ESTIMATED_TRADING_COST` (fraction; default `0.002` = 0.20% round-trip)

The backtest's fee/slippage inputs remain separate and are deducted from realized returns; its signal hurdle is round-trip cost plus the minimum net return. The main app's per-user saved preference remains an app-level setting; these worker variables configure the shared model-worker default, not an individual user's preference.

## Model and safety behavior

- The live model registry contains 72 short-term combinations: 12 analysis
  windows x 6 forecast horizons. Long-term analysis is handled exclusively by
  the separate `LongTerm-Torn-Helper-ML` repository.
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

## Minute-series quality and reconstruction

The short-term worker preserves raw provider observations in
stock_engine.prices and reconstructs only short gaps in memory when preparing
model features. It does not write estimated prices back into the raw table.

- One or two missing minutes: straight-line interpolation between the two
  genuine endpoint observations (for a single missing minute this is their
  arithmetic average).
- Three to five missing minutes: endpoint-anchored slope bridge. Every
  reconstructed point is explicitly marked synthetic in the worker's
  in-memory provenance mask.
- More than five missing minutes: never automatically filled. The gap remains
  unresolved and the existing feature-coverage checks can reject windows
  affected by it.

Training origins and future target labels must both be genuine observations.
Synthetic values may contribute to the surrounding feature window, but they
are excluded from supervised target labels. The ordinary backtest continues
to use the raw genuine series, so interpolated values cannot create realized
profits or improve reported test outcomes. The raw price cache also stores
only provider observations.

The cleaner is covered by unit tests for one-, two-, three-to-five-, and
greater-than-five-minute gaps, provenance flags, and invalid input. This
reconstruction is a feature-continuity aid, not recovered market truth.

## Backtesting (automatic two-hour retrigger loop)

`.github/workflows/backtest.yml` starts a separate retrigger job alongside the
backtest matrix. The timer waits two hours from the retrigger job's start, then uses the
GitHub Actions API to dispatch the next full backtest workflow on `main`. The workflow is started once manually; the retrigger loop
then continues automatically. GitHub runner queueing and dispatch latency mean
the interval is best-effort, not an exact two-hour start-to-start guarantee.
The workflow no longer uses a cron schedule.

The workflow requires the Actions write permission for its dispatch job. A
unit-test job runs once before the stock matrix begins. The matrix contains one
independent job for each of the 35 stocks, with up to eight stock jobs running
concurrently and fail-fast disabled so one stock's failure does not cancel the
others.

Each stock tests all 72 live short-term registry window/horizon combinations
(12 analysis windows x 6 forecast horizons) across all six supported backtest
model families. The scheduled defaults use 180 days of
history, all eligible out-of-sample history currently available in each shard
(up to the 180-day retention limit), and up to 30 walk-forward folds per
configuration. Forecast configurations are limited to the short-term registry; long-term forecasting is maintained in the dedicated repository.

Backtest predictions, trades, metrics and calibration candidates are stored
in the separate `stock_engine.ml_backtest_*` tables. Ridge-compatible
candidates can be promoted to live calibration only after at least 30 matured
live-forward checks and at least a 2% MAE improvement against the current
calibration. Other model families are comparison-only and are not wired into
the live predictor. No RLS policies are changed.

Each stock job has a 165-minute timeout. The workflow allows up to eight
concurrent stock jobs to limit database connection pressure. Workflow-level
concurrency prevents overlapping full batches; a dispatched run may wait for
the previous batch to finish.

The report job runs after the stock matrix and stores model rankings in
`stock_engine.ml_backtest_reports`. Before the matrix starts, the workflow now runs
`python/stock_ml/backtest_setup.py` across every configured shard. It validates
each shard assignment, applies the additive DDL in
`python/stock_ml/backtest_setup.sql`, and verifies that
`stock_engine.ml_backtest_reports` exists. This makes setup repeatable and
idempotent, and does not modify existing RLS policies.

## Backtest performance reporting

After the stock matrix finishes and only when schema provisioning succeeded,
the workflow runs `python/stock_ml/backtest_reporting.py` against each configured shard. It
evaluates completed runs from the preceding eight hours and stores a compact
JSON report in `stock_engine.ml_backtest_reports`. Reports include each
configuration's sample count, MAE, RMSE, zero-return baseline MAE, directional
accuracy, mean error and available trade results.

A configuration is eligible for ranking only when it has at least 30
out-of-sample predictions, at least five trades, positive net compounded
return, profit factor above 1.0, maximum drawdown no greater than 15%, and MAE
improves on the zero-return baseline by at least 2%. Eligible models are ranked
by net total return divided by the larger of absolute maximum drawdown or 1%,
then by profit factor, MAE improvement and directional accuracy. Models with
no trades, negative net returns or excessive drawdown are rejected rather than
being rewarded for low drawdown alone. The report records the strongest
eligible model per stock and per stock/window/horizon configuration, as well
as all evaluated rankings and risk metrics.
Insufficient samples and baseline underperformance are not presented as
successful models. These rankings are research outputs only: they do not
change the live predictor or automatically promote non-Ridge model families.
The report table is private and additive; installing the updated
`backtest_setup.sql` creates it without changing existing RLS policies.

## Tests

```sh
python -m pip install -r python/stock_ml/requirements.txt
python -m unittest discover -s python/stock_ml -v
```

No credentials, private holdings, account data or personal information belong
in this public repository.
