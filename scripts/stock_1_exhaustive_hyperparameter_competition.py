#!/usr/bin/env python3
"""
Torn Helper ML — exhaustive one-stock hyperparameter competition.

Requirements:
    Python 3.10+
    pip install "psycopg[binary]" numpy pandas scikit-learn tqdm joblib ipython

Run:
    python stock_1_exhaustive_hyperparameter_competition.py

Edit STOCK_ID below (1–35). The script reads the assigned PostgreSQL shard
using a read-only transaction and writes CSV/ZIP reports locally.
It does not modify the database or production model settings.
"""
# # Torn Helper ML — Exhaustive Hyperparameter Competition (one stock)
# 
# Run this notebook manually in Google Colab for **one stock at a time**. It tests every parameter combination in the declared grids for all six existing model families across all 36 short-term analysis-window/forecast-horizon selectors. It also searches ensemble weights on a 0.1 grid.
# 
# **This notebook is an offline research/backtesting tool.** It reads minute prices from the stock's assigned shard and writes CSV reports to the Colab runtime only. It does not write to Supabase/PostgreSQL, change production models, publish winners, or alter calibration. Backtesting measures prediction accuracy only: no fees, slippage, or trading-cost gates.
# 
# All trial hyperparameters are scored on chronological validation data inside each walk-forward fold. Only the chosen configuration for each family/selector is refit on the fold's full matured history and evaluated at the next untouched origin. This nested process avoids selecting a configuration on the same observation used for its reported out-of-sample score.

# ## 1. Install dependencies
# Run after starting a fresh Colab runtime.
# 

# ## 2. Choose stock and competition settings
# Stock assignment: stocks 1–2 use project 1, stocks 3–4 use project 2, etc. The query is strictly limited to the most recent 180 days. Increase `MAX_FOLDS` only if you want more walk-forward test origins; do not reduce the parameter grids if you want the exhaustive run.
# 

import os, json, math, time, hashlib, itertools, warnings
import multiprocessing
from joblib import Parallel, delayed, parallel_config
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from IPython.display import display


def main():

    STOCK_ID = 1                 # Change to 1–35; run one stock at a time
    LOOKBACK_DAYS = 180          # Short-term history retention boundary
    MAX_FOLDS = 30               # Walk-forward out-of-sample origins per selector
    MIN_TRAIN_ROWS = 200
    ORIGIN_STRIDE_MINUTES = 15
    WEIGHT_STEP = 0.1            # Exhaustive ensemble weight grid; weights sum to 1
    RANDOM_STATE = 17
    # Process workers are capped to avoid exhausting 32 GB RAM on large tree models.
    CPU_COUNT = multiprocessing.cpu_count()
    N_JOBS = max(1, min(8, CPU_COUNT - 1))
    OUTPUT_DIR = Path(__file__).resolve().parent / 'stock_exhaustive_results'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ANALYSIS_WINDOWS = (720, 1080, 1440, 4320, 10080, 43200)
    HORIZONS = (60, 180, 360, 720, 1080, 1440)
    SELECTORS = [(w, h) for w in ANALYSIS_WINDOWS for h in HORIZONS]

    if not 1 <= int(STOCK_ID) <= 35:
        raise ValueError('STOCK_ID must be between 1 and 35')
    if LOOKBACK_DAYS != 180:
        raise ValueError('This short-term competition is intentionally capped at 180 days of source data.')
    PROJECT_NUMBER = (int(STOCK_ID) + 1) // 2
    print(f'Stock {STOCK_ID} | shard/project {PROJECT_NUMBER} | {len(SELECTORS)} selectors')


    # ## 3. Connect to the correct shard and load the last 180 days
    # Use a database role with SELECT-only permissions if available. The session is also set to read-only. Never paste the DSN into a notebook cell or save it in the notebook.
    # 

    PG_DSN = input('PostgreSQL DSN for this stock shard: ').strip()
    print('Connecting to PostgreSQL...')
    conn = psycopg.connect(PG_DSN, sslmode='require', connect_timeout=30)
    print('PostgreSQL connected. Loading the last 180 days...')
    try:
        # Use a transaction-scoped read-only setting. This works with transaction-pooling
        # proxies that reset session settings between transactions.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute('SET TRANSACTION READ ONLY')  # Must be the first command in this transaction.
                cur.execute('SHOW transaction_read_only')
                if cur.fetchone()[0] != 'on':
                    raise RuntimeError('Could not start a read-only PostgreSQL transaction. Use a SELECT-only database role.')
                cur.execute('SELECT project_number, enabled FROM stock_engine.settings WHERE singleton')
                settings = cur.fetchone()
                if not settings or int(settings[0]) != PROJECT_NUMBER or not settings[1]:
                    raise RuntimeError('Shard assignment mismatch or shard is disabled.')
                cur.execute('SELECT EXISTS(SELECT 1 FROM stock_engine.stocks WHERE stock_id=%s)', (STOCK_ID,))
                if not cur.fetchone()[0]:
                    raise ValueError(f'Stock {STOCK_ID} is not present in this shard.')
                query = '''
                SELECT FLOOR(EXTRACT(EPOCH FROM observed_at)/60)::bigint AS minute_ts,
                       (ARRAY_AGG(price ORDER BY observed_at DESC))[1] AS price
                FROM stock_engine.prices
                WHERE stock_id = %s
                  AND observed_at >= NOW() - (%s * INTERVAL '1 day')
                GROUP BY 1
                ORDER BY 1
                '''
                cur.execute(query, (STOCK_ID, LOOKBACK_DAYS))
                rows = cur.fetchall()
                columns = [desc.name for desc in cur.description]
                price_df = pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()
    del PG_DSN
    if price_df.empty:
        raise RuntimeError('No price observations were found in the last 180 days.')
    times = price_df['minute_ts'].to_numpy(dtype=np.int64)
    prices = price_df['price'].to_numpy(dtype=np.float64)
    if len(times) != len(prices) or np.any(np.diff(times) <= 0) or np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError('Loaded minute price series failed validation.')
    print(f'Loaded {len(times):,} minute buckets; {len(times)/1440:.1f} observed days.')
    print(price_df.head().to_string(index=False))

    # ## 4. Define the exhaustive hyperparameter grids
    # Every Cartesian-product combination below is evaluated. No random search or early stopping of the search space is used. The HistGradientBoosting grid includes all specified values for `min_samples_leaf`.
    # 

    from itertools import product

    def make_grid(family):
        if family == 'ridge':
            return [{'alpha': a} for a in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000)]
        if family == 'elastic_net':
            return [{'alpha': a, 'l1_ratio': r}
                    for a in (0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01)
                    for r in (0.05, 0.15, 0.25, 0.5, 0.75, 0.9, 0.95)]
        if family in ('random_forest', 'extra_trees'):
            return [dict(n_estimators=n, max_depth=d, min_samples_leaf=m, max_features=f)
                    for n in (100, 300, 500, 800)
                    for d in (None, 4, 6, 8, 12, 16)
                    for m in (1, 2, 4, 8, 16)
                    for f in (0.5, 0.75, 1.0, 'sqrt')]
        if family == 'hist_gradient_boosting':
            return [dict(max_iter=i, max_leaf_nodes=l, learning_rate=lr,
                         l2_regularization=l2, min_samples_leaf=m)
                    for i in (100, 200, 400, 800)
                    for l in (7, 15, 31, 63, 127)
                    for lr in (0.01, 0.03, 0.05, 0.1)
                    for l2 in (0, 0.1, 1, 5, 10)
                    for m in (10, 20, 40)]
        raise ValueError(f'Unknown family: {family}')

    FAMILIES = ('ridge', 'elastic_net', 'random_forest', 'extra_trees', 'hist_gradient_boosting')
    GRIDS = {family: make_grid(family) for family in FAMILIES}
    grid_counts = {family: len(grid) for family, grid in GRIDS.items()}
    assert grid_counts == {'ridge': 9, 'elastic_net': 49, 'random_forest': 480,
                           'extra_trees': 480, 'hist_gradient_boosting': 1200}
    BASE_TRIALS = sum(grid_counts.values())
    print('Parameter configurations per selector:', grid_counts)
    print(f'Base configurations per selector: {BASE_TRIALS:,}')
    print(f'Base configurations across 36 selectors: {BASE_TRIALS * len(SELECTORS):,}')
    print('Ensemble: all five-family simplex weights in 0.1 increments, plus mean and median.')


    # ## 5. Causal features and purged walk-forward folds
    # Features reproduce the short-term engine's current feature construction. Labels are admitted to a fold's training data only after their exact target minute has matured. Test origins are at least one forecast horizon apart, so test outcomes do not overlap. The final test origin is never used to choose hyperparameters.
    # 

    def exact_price(ts, px, queries):
        q = np.asarray(queries, dtype=np.int64)
        idx = np.searchsorted(ts, q)
        clipped = np.minimum(idx, len(ts)-1)
        return np.where((idx < len(ts)) & (ts[clipped] == q), px[clipped], np.nan)

    def feature_matrix(ts, px, origins, window):
        logs = np.log(px)
        sums = np.r_[0.0, np.cumsum(logs)]
        squares = np.r_[0.0, np.cumsum(logs*logs)]
        adjacent = np.r_[0.0, np.where(np.diff(ts)==1, np.diff(logs)**2, 0.0)]
        counts = np.r_[0, (np.diff(ts)==1).astype(int)]
        rv = np.r_[0.0, np.cumsum(adjacent)]
        rc = np.r_[0, np.cumsum(counts)]
        left = np.searchsorted(ts, origins-window, side='right')
        right = np.searchsorted(ts, origins, side='right')
        count = right-left
        mean = (sums[right]-sums[left]) / np.maximum(count, 1)
        variance = np.maximum(0.0, (squares[right]-squares[left])/np.maximum(count,1)-mean*mean)
        current = exact_price(ts, px, origins)
        lags = [min(window-1, lag) for lag in (1,5,15,60,240,1440)]
        changes = [current/exact_price(ts, px, origins-lag)-1 for lag in lags]
        first_return = np.minimum(left+1, right)
        volatility = np.sqrt((rv[right]-rv[first_return]) / np.maximum(1, rc[right]-rc[first_return]))
        x = np.column_stack([*changes, np.log(current)-mean, np.sqrt(variance), volatility, count/window])
        first = ts[np.minimum(left, len(ts)-1)]
        valid = ((count >= np.ceil(window*0.95)) &
                 (first <= origins-window+max(1, window//100)) & np.all(np.isfinite(x), axis=1))
        return x, valid

    def build_folds(ts, px, window, horizon, max_folds=MAX_FOLDS):
        origins = ts[ts % ORIGIN_STRIDE_MINUTES == 0]
        if not len(origins): return []
        x_all, feature_ok = feature_matrix(ts, px, origins, window)
        target = exact_price(ts, px, origins+horizon)
        current = exact_price(ts, px, origins)
        matured = feature_ok & np.isfinite(target) & np.isfinite(current) & (current > 0)
        returns = np.full(len(origins), np.nan, dtype=float)
        returns[matured] = target[matured]/current[matured]-1.0
        candidates = origins[origins >= ts[0]+window+horizon+MIN_TRAIN_ROWS*ORIGIN_STRIDE_MINUTES]
        if len(candidates) > max_folds:
            candidates = candidates[np.linspace(0, len(candidates)-1, max_folds).astype(int)]
        nonoverlap, last_exit = [], -1
        for c in candidates:
            if int(c) >= last_exit:
                nonoverlap.append(int(c)); last_exit = int(c)+horizon
        mi = np.flatnonzero(matured)
        target_times = origins[mi]+horizon
        folds=[]
        for origin in nonoverlap:
            eligible = int(np.searchsorted(target_times, origin, side='right'))
            if eligible < 240: continue
            rows = mi[:eligible]
            oi = int(np.searchsorted(origins, origin))
            if oi >= len(origins) or origins[oi] != origin or not feature_ok[oi]: continue
            historical_origins = origins[rows]
            boundary = historical_origins[int(len(historical_origins)*0.75)]
            fit_idx = rows[historical_origins+horizon < boundary]
            val_idx = rows[historical_origins >= boundary]
            if len(fit_idx)<100 or len(val_idx)<30: continue
            folds.append(dict(origin=origin, x=x_all, y=returns, fit_idx=fit_idx,
                              val_idx=val_idx, full_idx=rows, eval_idx=oi,
                              origin_price=float(current[oi]), actual_price=float(target[oi]),
                              actual_return=float(target[oi]/current[oi]-1.0)))
        return folds

    print('Fold builder ready. Each selector will report its actual usable fold count.')

    # ## 6. Estimator factory and exhaustive ensemble weight grid
    # For each fold, all 2,218 base configurations are scored on the same validation observations. The best validation configuration within each base family is then refit on the complete matured history. The ensemble search evaluates all nonnegative five-model weight combinations summing to 1 in increments of 0.1 (1,001 combinations), plus median and equal-weight mean. The ensemble is selected on validation predictions and then evaluated on the untouched fold origin.
    # 

    def fit_predict(family, params, x_train, y_train, x_eval):
        if family in ('ridge','elastic_net'):
            scaler = StandardScaler().fit(x_train)
            z_train, z_eval = scaler.transform(x_train), scaler.transform(x_eval)
            if family == 'ridge':
                model = Ridge(**params)
            else:
                model = ElasticNet(**params, max_iter=10000, tol=1e-5, selection='cyclic')
            model.fit(z_train, y_train)
            return model.predict(z_eval)
        if family == 'random_forest':
            model = RandomForestRegressor(**params, n_jobs=1, random_state=RANDOM_STATE)
        elif family == 'extra_trees':
            model = ExtraTreesRegressor(**params, n_jobs=1, random_state=RANDOM_STATE)
        elif family == 'hist_gradient_boosting':
            model = HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        else:
            raise ValueError(family)
        model.fit(x_train, y_train)
        return model.predict(x_eval)

    def compositions(total, parts):
        if parts == 1:
            yield (total,); return
        for first in range(total+1):
            for rest in compositions(total-first, parts-1):
                yield (first,)+rest

    WEIGHT_VECTORS = np.asarray(list(compositions(10, 5)), dtype=float)/10.0
    assert WEIGHT_VECTORS.shape == (1001, 5)
    print('Ensemble weight vectors:', len(WEIGHT_VECTORS))

    # ## 7. Run the exhaustive competition
    # This is computationally intensive by design. The notebook checkpoints completed fold results after every fold, so an interrupted run can retain completed work. Rerunning starts the competition again and overwrites the output files for this stock. Keep the Colab runtime active until completion.
    # 

    def config_id(family, params):
        raw = json.dumps({'family':family, 'params':params}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def evaluate_config(family, params, x_fit, y_fit, x_val, y_val,
                        window, horizon, fold_number, origin):
        """Run one independent hyperparameter trial in a separate worker process."""
        try:
            pred = fit_predict(family, params, x_fit, y_fit, x_val)
            score = float(np.mean(np.abs(y_val-pred)))
            row={'stock_id':STOCK_ID,'analysis_minutes':window,'horizon_minutes':horizon,
                 'fold':fold_number,'origin_minute':int(origin),
                 'family':family,'config_id':config_id(family,params),
                 'params_json':json.dumps(params,sort_keys=True,default=str),
                 'validation_mae':score,'validation_rmse':float(np.sqrt(np.mean((y_val-pred)**2))),
                 'validation_directional_accuracy':float(np.mean((pred>=0)==(y_val>=0))),
                 'validation_samples':len(y_val),'status':'ok'}
            return row, pred, score
        except Exception as exc:
            row={'stock_id':STOCK_ID,'analysis_minutes':window,'horizon_minutes':horizon,
                 'fold':fold_number,'origin_minute':int(origin),'family':family,
                 'config_id':config_id(family,params),
                 'params_json':json.dumps(params,sort_keys=True,default=str),
                 'validation_mae':np.nan,'validation_rmse':np.nan,
                 'validation_directional_accuracy':np.nan,'validation_samples':len(y_val),
                 'status':'error: '+repr(exc)[:300]}
            return row, None, float('inf')


    def run_one_fold(fold, selector, fold_number):
        window, horizon = selector
        x, y = fold['x'], fold['y']
        fit_idx, val_idx, full_idx = fold['fit_idx'], fold['val_idx'], fold['full_idx']
        x_fit, y_fit = x[fit_idx], y[fit_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        x_full, y_full = x[full_idx], y[full_idx]
        x_test = x[fold['eval_idx']:fold['eval_idx']+1]
        validation_predictions = {}
        trial_rows=[]
        best_by_family={}

        # loky uses separate Python processes. Limit each process to one native
        # math/BLAS/OpenMP thread to prevent nested oversubscription.
        with parallel_config(backend='loky', inner_max_num_threads=1):
            for family in FAMILIES:
                print(f'  {family}: evaluating {len(GRIDS[family])} configurations '
                      f'using {N_JOBS} processes...', flush=True)
                results = Parallel(
                    n_jobs=N_JOBS,
                    batch_size=1,
                    pre_dispatch=N_JOBS,
                    max_nbytes='10M',
                    mmap_mode='r'
                )(
                    delayed(evaluate_config)(
                        family, params, x_fit, y_fit, x_val, y_val,
                        window, horizon, fold_number, fold['origin']
                    )
                    for params in GRIDS[family]
                )

                best_score=float('inf'); best_params=None; best_val_pred=None
                for params, (row, pred, score) in zip(GRIDS[family], results):
                    trial_rows.append(row)
                    if pred is not None and score < best_score:
                        best_score, best_params, best_val_pred = score, params, pred

                if best_params is None:
                    raise RuntimeError(f'Every {family} trial failed for selector {selector}, fold {fold_number}.')
                validation_predictions[family]=best_val_pred
                best_by_family[family]={'params':best_params,'validation_mae':best_score,
                                        'config_id':config_id(family,best_params)}

        # Exhaustively select the ensemble using validation data only.
        val_matrix=np.column_stack([validation_predictions[f] for f in FAMILIES])
        best_ensemble_score=float('inf'); best_weights=None; best_ensemble_kind=None
        for weights in WEIGHT_VECTORS:
            pred=val_matrix @ weights
            score=float(np.mean(np.abs(y_val-pred)))
            if score < best_ensemble_score:
                best_ensemble_score, best_weights, best_ensemble_kind = score, weights.copy(), 'weighted'
        for kind, pred in [('mean',np.mean(val_matrix,axis=1)),('median',np.median(val_matrix,axis=1))]:
            score=float(np.mean(np.abs(y_val-pred)))
            if score < best_ensemble_score:
                best_ensemble_score, best_weights, best_ensemble_kind = score, None, kind

        # Refit only each selected family configuration on all matured rows before test origin.
        test_predictions={}
        for family in FAMILIES:
            params=best_by_family[family]['params']
            test_predictions[family]=float(fit_predict(family,params,x_full,y_full,x_test)[0])
        test_matrix=np.asarray([test_predictions[f] for f in FAMILIES],dtype=float)
        if best_ensemble_kind=='weighted': ensemble_test=float(test_matrix @ best_weights)
        elif best_ensemble_kind=='mean': ensemble_test=float(np.mean(test_matrix))
        else: ensemble_test=float(np.median(test_matrix))

        prediction_rows=[]
        for family in FAMILIES:
            prediction_rows.append({'stock_id':STOCK_ID,'analysis_minutes':window,'horizon_minutes':horizon,
                'fold':fold_number,'origin_minute':int(fold['origin']),
                'target_minute':int(fold['origin']+horizon),'family':family,
                'config_id':best_by_family[family]['config_id'],
                'params_json':json.dumps(best_by_family[family]['params'],sort_keys=True,default=str),
                'predicted_return':test_predictions[family],'actual_return':fold['actual_return'],
                'absolute_error':abs(fold['actual_return']-test_predictions[family]),
                'direction_correct':bool((test_predictions[family]>=0)==(fold['actual_return']>=0)),
                'validation_mae':best_by_family[family]['validation_mae']})
        prediction_rows.append({'stock_id':STOCK_ID,'analysis_minutes':window,'horizon_minutes':horizon,
            'fold':fold_number,'origin_minute':int(fold['origin']),'target_minute':int(fold['origin']+horizon),
            'family':'ensemble','config_id':best_ensemble_kind,
            'params_json':json.dumps({'kind':best_ensemble_kind,'weights':dict(zip(FAMILIES,best_weights.tolist()))
                                      if best_weights is not None else None},sort_keys=True),
            'predicted_return':ensemble_test,'actual_return':fold['actual_return'],
            'absolute_error':abs(fold['actual_return']-ensemble_test),
            'direction_correct':bool((ensemble_test>=0)==(fold['actual_return']>=0)),
            'validation_mae':best_ensemble_score})
        return trial_rows, prediction_rows, best_by_family, {
            'analysis_minutes':window,'horizon_minutes':horizon,'fold':fold_number,
            'origin_minute':int(fold['origin']),'ensemble_kind':best_ensemble_kind,
            'ensemble_weights':json.dumps(dict(zip(FAMILIES,best_weights.tolist())) if best_weights is not None else None),
            'ensemble_validation_mae':best_ensemble_score}

    all_predictions=[]; all_selections=[]; fold_errors=[]
    TRIALS_PATH=OUTPUT_DIR/'validation_trials_checkpoint.csv'
    if TRIALS_PATH.exists(): TRIALS_PATH.unlink()  # Fresh run for this stock
    progress_total=len(SELECTORS)
    print(f'CPU cores detected: {CPU_COUNT}; multiprocessing workers: {N_JOBS}', flush=True)
    for selector_idx, selector in enumerate(tqdm(SELECTORS, desc='Selectors')):
        window,horizon=selector
        folds=build_folds(times,prices,window,horizon,MAX_FOLDS)
        print(f'\nSelector {selector_idx+1}/{progress_total}: window={window}m horizon={horizon}m folds={len(folds)}', flush=True)
        if not folds:
            fold_errors.append({'analysis_minutes':window,'horizon_minutes':horizon,'error':'insufficient_history_or_folds'})
            continue
        for fold_number, fold in enumerate(tqdm(folds,desc=f'{window}m/{horizon}m folds',leave=False),start=1):
            try:
                trial_rows,pred_rows,best,ens=run_one_fold(fold,selector,fold_number)
                pd.DataFrame(trial_rows).to_csv(TRIALS_PATH,mode='a',header=not TRIALS_PATH.exists(),index=False)
                all_predictions.extend(pred_rows); all_selections.append(ens)
                for family, choice in best.items():
                    all_selections.append({'analysis_minutes':window,'horizon_minutes':horizon,
                        'fold':fold_number,'family':family,'config_id':choice['config_id'],
                        'params_json':json.dumps(choice['params'],sort_keys=True,default=str),
                        'validation_mae':choice['validation_mae']})
            except Exception as exc:
                fold_errors.append({'analysis_minutes':window,'horizon_minutes':horizon,
                                    'fold':fold_number,'error':repr(exc)[:1000]})
            # Checkpoint after every fold to protect long-running PC sessions.
            # Trial rows are streamed to disk after each fold to keep Colab RAM bounded.
            pd.DataFrame(all_predictions).to_csv(OUTPUT_DIR/'oos_predictions_checkpoint.csv',index=False)
            pd.DataFrame(all_selections).to_csv(OUTPUT_DIR/'selected_configs_checkpoint.csv',index=False)
            pd.DataFrame(fold_errors).to_csv(OUTPUT_DIR/'fold_errors_checkpoint.csv',index=False)

    print('\nCompetition loop finished.')
    trial_count = sum(1 for _ in open(TRIALS_PATH, 'r', encoding='utf-8')) - 1 if TRIALS_PATH.exists() else 0
    print('Successful OOS predictions:',len(all_predictions),'| trial scores:',trial_count,'| fold errors:',len(fold_errors))

    # ## 8. Build leaderboards and export reports
    # The validation leaderboard ranks individual parameter configurations by their validation scores across folds. The out-of-sample leaderboard evaluates the nested selected model at untouched forecast origins. Do not confuse the validation leaderboard with final OOS accuracy.
    # 

    trials=pd.read_csv(TRIALS_PATH) if TRIALS_PATH.exists() else pd.DataFrame()
    preds=pd.DataFrame(all_predictions)
    selections=pd.DataFrame(all_selections)
    errors_df=pd.DataFrame(fold_errors)

    if not trials.empty:
        validation_leaderboard=(trials[trials.status=='ok']
            .groupby(['analysis_minutes','horizon_minutes','family','config_id','params_json'],dropna=False)
            .agg(folds_scored=('validation_mae','count'),mean_validation_MAE=('validation_mae','mean'),
                 median_validation_MAE=('validation_mae','median'),mean_validation_RMSE=('validation_rmse','mean'),
                 mean_validation_directional_accuracy=('validation_directional_accuracy','mean'))
            .reset_index().sort_values(['analysis_minutes','horizon_minutes','family','mean_validation_MAE']))
    else:
        validation_leaderboard=pd.DataFrame()

    if not preds.empty:
        def summarize_oos(g):
            err=g['actual_return']-g['predicted_return']
            return pd.Series({'samples':len(g),'MAE_pct':np.mean(np.abs(err))*100,
                'RMSE_pct':np.sqrt(np.mean(err**2))*100,
                'directional_accuracy_pct':np.mean(g['direction_correct'])*100,
                'mean_prediction_bias_pct':np.mean(g['predicted_return']-g['actual_return'])*100,
                'correlation':g['predicted_return'].corr(g['actual_return']) if len(g)>1 else np.nan})
        oos_leaderboard=(preds.groupby(['analysis_minutes','horizon_minutes','family'])
                         .apply(summarize_oos,include_groups=False).reset_index()
                         .sort_values(['analysis_minutes','horizon_minutes','MAE_pct']))
    else:
        oos_leaderboard=pd.DataFrame()

    if not selections.empty:
        selected_configs=selections[selections.get('family',pd.Series(index=selections.index,dtype=object)).notna()].copy()
        selection_frequency=(selected_configs.groupby(['analysis_minutes','horizon_minutes','family','config_id','params_json'],dropna=False)
            .agg(selected_folds=('fold','count'),mean_selected_validation_MAE=('validation_mae','mean'))
            .reset_index().sort_values(['analysis_minutes','horizon_minutes','family','selected_folds'],ascending=[True,True,True,False]))
    else:
        selection_frequency=pd.DataFrame()

    prefix=f'stock_{STOCK_ID}_exhaustive'
    files={
     'validation_trials':OUTPUT_DIR/f'{prefix}_all_validation_trials.csv',
     'validation_leaderboard':OUTPUT_DIR/f'{prefix}_validation_leaderboard.csv',
     'oos_predictions':OUTPUT_DIR/f'{prefix}_oos_predictions.csv',
     'oos_leaderboard':OUTPUT_DIR/f'{prefix}_oos_leaderboard.csv',
     'selected_configs':OUTPUT_DIR/f'{prefix}_selected_configurations.csv',
     'fold_errors':OUTPUT_DIR/f'{prefix}_fold_errors.csv'}
    for key,path in files.items():
        frame={'validation_trials':trials,'validation_leaderboard':validation_leaderboard,
               'oos_predictions':preds,'oos_leaderboard':oos_leaderboard,
               'selected_configs':selection_frequency,'fold_errors':errors_df}[key]
        frame.to_csv(path,index=False)
    print('Saved reports:')
    for k,p in files.items(): print(k, '->', p)
    print((oos_leaderboard.head(30) if not oos_leaderboard.empty else oos_leaderboard).to_string(index=False))

    # ## 9. Download all reports
    # The ZIP contains every trial's validation scores, selected configurations, untouched OOS predictions, aggregate leaderboards, and any fold errors.
    # 

    import zipfile
    zip_path=OUTPUT_DIR/f'stock_{STOCK_ID}_exhaustive_competition_reports.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files.values():
            if p.exists(): zf.write(p,arcname=p.name)
    print(f'\nAll reports packaged at: {zip_path.resolve()}')

    # ## 10. Interpretation and limitations
    # - The 2,218 base configurations are tested independently for every one of the 36 selectors: 79,848 base configurations per stock before ensemble search.
    # - The ensemble grid contains 1,001 five-model weight vectors (all nonnegative weights in 0.1 increments summing to 1), plus mean and median. It uses the best validation-selected configuration from each base family within that fold.
    # - All feature and label rows are causal; target labels are included in training only after their exact target minute has passed.
    # - The validation leaderboard is a tuning diagnostic. The untouched-origin OOS leaderboard is the more important generalization check.
    # - Repeatedly selecting the best from thousands of candidates can still overfit validation data. Use a later untouched period or a second independent evaluation before production promotion.
    # - This notebook does not automatically install winners into Torn Helper ML or update calibration. Production integration must be a separate reviewed change.
    # - If Colab disconnects, checkpoint CSVs remain in the runtime only while that runtime survives. Download the ZIP after completion; a runtime reset loses local files.
    # - Backtesting is accuracy-only. No trading costs, fees, slippage, profit thresholds, or trading signals are used.
    # 
    # Close the notebook when finished. Run it again with a different `STOCK_ID` to evaluate another stock.


if __name__ == '__main__':
    main()
