# Torn-Helper ML

Public Python research engine for Torn-Helper. The private application remains in
[unnyanr11/Torn-Helper](https://github.com/unnyanr11/Torn-Helper).

## Backtesting & calibration

The walk-forward engine in `python/stock_ml/backtesting.py` replays actual
historical observations. Each fold trains on a strict chronological prefix ending
at its decision time; future prices are used only to score the matured prediction.
It reports prediction error, return, win rate, drawdown, fees, slippage and virtual
capital. It does not execute real trades.

Backtest calibration is a **candidate**, not an unconditional live-model change.
A correction is fitted on the first 70% of chronological predictions and evaluated
on the remaining 30%. It is accepted only if holdout MAE improves. Production
promotion must additionally pass a later, untouched forward-validation window.
Failed candidates leave the active live calibration unchanged.

## Setup

- Python 3.12
- Install `pip install -r python/stock_ml/requirements.txt`
- Run `python -m unittest discover -s python/stock_ml -v`
- Install `python/stock_ml/backtest_setup.sql` in each assigned stock shard before persisting results.
- Keep all PostgreSQL URLs in GitHub Actions secrets, never in source.

The source engine uses real observations and does not fill missing timestamps.
Results are limited by available history and its cadence. A quarter-hour dataset
cannot support minute-accurate fills. The current simulator assumes entry at the
decision close and exit at the exact horizon close; actual execution, queue priority,
spread and market impact can differ.

## Security

No Supabase credentials, API keys, user portfolios or private application data
belong in this public repository. Only public market-price observations and
research code are suitable for publication.
