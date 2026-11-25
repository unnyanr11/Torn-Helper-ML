# Torn-Helper ML

Backend-only machine-learning and research worker for Torn-Helper. There is no
website backtesting interface: scheduled jobs evaluate the models in the background
and store results/calibration in the assigned Supabase stock shards.

## Autonomous learning cycle

- Live prediction workers continue to train and publish forecasts on their existing cadence.
- The model performs nested chronological Ridge hyperparameter selection over
  alpha values 1, 10, 100 and 1000. The selection window is separate from the
  final untouched validation window, with a forecast-horizon purge between them.
- A GitHub Actions job runs every two hours against actual stored prices. Each
  stock is evaluated across 24 combinations: three feature windows (1, 3 and 7
  days), four forecast horizons (1, 3, 6 and 24 hours), and both short/long
  model styles. Each combination uses up to 12 non-overlapping evaluation folds
  per stock to keep the two-hour workload bounded.
- Backtest results, prediction outcomes, trade assumptions and calibration
  candidates are written to private `stock_engine` tables.
- A calibration candidate is not promoted based on backtest results alone. It
  must improve MAE by at least 2% on at least 30 matured live forward outcomes.
  Otherwise the current calibration remains in place.
- The existing live prediction engine reads the approved calibration payload.
  No user-facing dashboard or manual backtesting action is required.

## Required setup

1. Add the repository Actions secret `STOCK_ML_PROJECTS_JSON`, containing a JSON
   object mapping configured shard numbers (1-18) to PostgreSQL session-pooler
   connection URLs. Never commit connection URLs or credentials.
2. In each shard, install the existing live ML schema first (including
   `stock_engine.ml_v2_checks` and `stock_engine.ml_v2_calibration`), then run
   `python/stock_ml/backtest_setup.sql`. The SQL creates isolated backtest
   result tables and does not alter existing RLS policies.
3. Ensure the database role used by the secret can read `stock_engine.prices`
   and write the new private `stock_engine.ml_backtest_*` tables and the
   existing ML calibration table.
4. Enable GitHub Actions and check the first manual run before relying on the
   daily schedule.

The scheduled workflow can be manually dispatched for a selected set of stock
IDs or a 30-180 day lookback. The default workload is capped at 12 folds per
configuration to keep the 24-combination run bounded.

## Backtesting assumptions and limitations

The simulator uses actual recorded prices, but assumes entry at the decision
minute close and exit at the exact horizon close. Fees and slippage are explicit
assumptions; real spreads, order-book depth, market impact and fill timing may
differ. Minute observations are deduplicated to the last observation per minute.
Missing observations are not fabricated. Insufficient history produces no
candidate rather than a forced calibration.

Backtesting is research, not proof of profitability. The backtest calibration
candidate is fitted on a chronological subset and checked on a later holdout;
promotion additionally requires independent matured live forward outcomes.
The current matrix varies feature-window length, forecast horizon and short/long
model style. It still uses the Ridge model family; it does not yet compare
fundamentally different algorithm families such as random forests or gradient
boosting. A broader family comparison should be added only with equivalent
leakage-safe selection and holdout validation.

## Tests

```sh
python -m pip install -r python/stock_ml/requirements.txt
python -m unittest discover -s python/stock_ml -v
```

No credentials, private holdings, account data or personal information belong in
this public repository.
