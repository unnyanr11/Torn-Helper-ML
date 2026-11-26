# Stock-specific adaptive algorithm selection

The short-term prediction engine uses a stable numbered algorithm registry:

| ID | Family |
|---:|---|
| 1 | Ridge |
| 2 | ElasticNet |
| 3 | Random Forest |
| 4 | Extra Trees |
| 5 | HistGradientBoosting |
| 6 | Ensemble |

IDs are permanent. Add new families with new IDs; never renumber or reuse an existing ID.

The backtest publisher chooses a winner independently for every stock, analysis window, and forecast horizon using the existing leakage-safe walk-forward comparison and accuracy-only acceptance criteria. The selected family is refit on eligible history and published to that stock shard's `stock_engine.ml_v2_models` row. The model payload includes `algorithm_id`, `model_family`, and a unique `model_id`.

The distributed backtest batch workflow runs the publisher after its runner succeeds, scoped to the batch's stock IDs. The live Python prediction worker loads that stock/configuration's published model and keeps it authoritative over routine fallback retraining. Prediction results include the selected algorithm ID and family.

A stock/configuration with no eligible winner keeps its previous published winner; it does not get replaced by a weaker or unvalidated candidate. Backtest scoring remains accuracy-only, without fees, slippage, trading simulation, or net-profit gates. Live Buy/Sell signal rules and costs remain separate and unchanged.

Long-term weekly forecasts are handled by the separate `LongTerm-Torn-Helper-ML` repository and its corresponding algorithm registry.
