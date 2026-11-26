# Distributed backtesting batches

The 35 stock IDs are split into nine manually dispatched GitHub Actions workflows:

| Batch | Stock IDs |
|---|---|
| 01 | 1, 2, 3, 4 |
| 02 | 5, 6, 7, 8 |
| 03 | 9, 10, 11, 12 |
| 04 | 13, 14, 15, 16 |
| 05 | 17, 18, 19, 20 |
| 06 | 21, 22, 23, 24 |
| 07 | 25, 26, 27, 28 |
| 08 | 29, 30, 31, 32 |
| 09 | 33, 34, 35 |

## Running a batch

1. Fork this repository into the GitHub account that will provide the runner.
2. In that fork, open **Settings → Secrets and variables → Actions**.
3. Add the required repository secret named `STOCK_ML_PROJECTS_JSON`.
4. Open **Actions**, select the desired **Backtest batch NN** workflow, and choose **Run workflow**.
5. Keep the default settings unless the maintainer has asked you to change them.

Each batch invokes the existing `python/stock_ml/backtest_runner.py` with a comma-separated `STOCK_IDS` list. The defaults remain a 180-day lookback, all eligible evaluation history, up to 30 walk-forward folds, and 2 Python workers. Calibration promotion is disabled.

## Database credential safety

The workflow needs database connection information through `STOCK_ML_PROJECTS_JSON`. Do not commit this value to the repository, put it in workflow YAML, or share it in logs or issues. A fork owner must only receive credentials intentionally provisioned for that participant.

**Do not distribute a Supabase owner, service-role, or unrestricted database credential to untrusted contributors.** Before inviting external participants, provision a dedicated least-privilege database role or a controlled result-submission API. The current runner writes backtest run/results to the configured database, so a read-only price-data credential alone is not sufficient for the unmodified runner. Restrict write access to the backtest tables and only the required stock/shard data; do not change RLS policies as part of this workflow change.

These workflows are manually dispatched. They do not automatically assign jobs to volunteers, coordinate a central queue, or validate results submitted by a participant. Avoid dispatching the same batch from multiple forks at the same time unless duplicate work is intended.
