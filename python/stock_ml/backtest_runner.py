"""Run walk-forward backtests against each configured shard's observed prices."""
import json, os, sys, uuid, traceback
import psycopg
import numpy as np
from psycopg.types.json import Jsonb
from backtesting import run_backtest, calibration_from_backtest, MODEL_FAMILIES
from realtime_model import COMBINATIONS
from strategy_engine import run_strategy_competition

# Exercise the complete live registry: 12 short windows x 6 horizons and
# 8 long windows x 4 horizons (104 total). Compare all supported backtest
# families; only Ridge-compatible candidates can ever be considered for live use.
BACKTEST_COMBINATIONS = COMBINATIONS
BACKTEST_MODEL_FAMILIES = MODEL_FAMILIES


def validate_and_promote(conn, stock, run_id, candidate, *, style, window_minutes, horizon_minutes):
    # Never promote on backtest performance alone. Use later live forward outcomes.
    if not candidate.get('active'):
        return dict(promoted=False, reason=candidate.get('reason', 'backtest_candidate_rejected'))
    rows = conn.execute("""
        select raw_return,predicted_return,actual_return
        from stock_engine.ml_v2_checks
        where stock_id=%s and style=%s and analysis_minutes=%s
          and horizon_minutes=%s and checked_at is not null
          and target_at<=now() and target_at>=now()-interval '90 days'
        order by target_at desc limit 120
    """, (stock,style,window_minutes,horizon_minutes)).fetchall()
    if len(rows) < 30:
        return dict(promoted=False, reason='insufficient_live_forward_samples', samples=len(rows))
    raw=np.asarray([float(r[0]) for r in rows])
    current=np.asarray([float(r[1]) for r in rows])
    actual=np.asarray([float(r[2]) for r in rows])
    corrected=candidate['scale']*raw+candidate['bias']
    baseline_mae=float(np.mean(np.abs(actual-current)))
    candidate_mae=float(np.mean(np.abs(actual-corrected)))
    if not candidate_mae < baseline_mae*0.98:
        return dict(promoted=False, reason='live_forward_holdout_did_not_improve_by_2pct',
                    samples=len(rows), baseline_mae=baseline_mae, candidate_mae=candidate_mae)
    q90=float(np.quantile(np.abs(actual-corrected), .90))
    payload=dict(scale=float(candidate['scale']), bias=float(candidate['bias']),
        error_q90=q90, samples=len(rows), active=True,
        source='walk_forward_backtest_plus_live_forward_validation',
        source_backtest_run=str(run_id), live_forward_mae=candidate_mae,
        previous_mae=baseline_mae)
    conn.execute("""
        insert into stock_engine.ml_v2_calibration
        (stock_id,style,analysis_minutes,horizon_minutes,payload)
        values(%s,%s,%s,%s,%s)
        on conflict(stock_id,style,analysis_minutes,horizon_minutes)
        do update set payload=excluded.payload,calibrated_at=now()
    """, (stock,style,window_minutes,horizon_minutes,Jsonb(payload)))
    conn.execute("""
        update stock_engine.ml_backtest_calibration_candidates
        set promoted=true,promotion_reason=%s where run_id=%s
    """, (f'live_forward_validated: {baseline_mae:.8g} -> {candidate_mae:.8g}', run_id))
    return dict(promoted=True, samples=len(rows), baseline_mae=baseline_mae,
                candidate_mae=candidate_mae, error_q90=q90)

def process(project,dsn,stock_ids,lookback_days,evaluation_days,max_folds):
    with psycopg.connect(dsn,sslmode='require',connect_timeout=15,autocommit=True) as conn:
        settings=conn.execute("select project_number,enabled from stock_engine.settings where singleton").fetchone()
        if not settings or int(settings[0])!=project or not settings[1]:
            raise ValueError('project assignment mismatch or disabled project')
        stocks=[int(r[0]) for r in conn.execute('select stock_id from stock_engine.stocks order by stock_id')]
        for stock in stocks:
            if stock not in (project*2-1,project*2) or stock>35: raise ValueError('unexpected stock assignment')
            if stock_ids and stock not in stock_ids: continue
            rows=conn.execute("""select floor(extract(epoch from observed_at)/60)::bigint,
                       (array_agg(price order by observed_at desc))[1]
                from stock_engine.prices where stock_id=%s and observed_at>=now()-(%s * interval '1 day')
                group by 1 order by 1""",(stock,lookback_days)).fetchall()
            if len(rows)<1000: print(f"stock {stock}: insufficient raw observations"); continue
            times=[int(r[0]) for r in rows]; prices=[float(r[1]) for r in rows]
            # Default to all eligible history available in this shard. Each fold
            # trains only on observations strictly before its own origin; run_backtest
            # also enforces the window, horizon and minimum-training requirements.
            # An explicit positive evaluation_days value remains available for
            # controlled experiments, but scheduled runs use 0 = all available.
            evaluation_start=(int(times[-1]-evaluation_days*1440) if evaluation_days>0 else None)
            available_days=max(0.0,(times[-1]-times[0])/1440.0)
            combinations=BACKTEST_COMBINATIONS
            for style,window_minutes,horizon_minutes in combinations:
              for model_family in BACKTEST_MODEL_FAMILIES:
                        run_id=uuid.uuid4()
                        config=dict(style=style,window_minutes=window_minutes,horizon_minutes=horizon_minutes,
                            fee_bps=10,slippage_bps=5,initial_capital=10000,max_folds=max_folds,
                            lookback_days=lookback_days,evaluation_days=(evaluation_days or None),
                            evaluation_mode=('all_available_after_minimum_training' if evaluation_days==0 else 'fixed_recent_window'),
                            available_history_days=round(available_days,2),
                            evaluation_start_minute=evaluation_start,
                            execution='decision-close to exact-horizon close',
                            combination_count=len(combinations),model_family_count=len(BACKTEST_MODEL_FAMILIES),model_family=model_family)
                        conn.execute("insert into stock_engine.ml_backtest_runs(run_id,stock_id,status,config,source_commit) values(%s,%s,'running',%s,%s)",
                            (run_id,stock,Jsonb(config),os.getenv('GITHUB_SHA')))
                        result=run_backtest(times,prices,window_minutes=window_minutes,
                            horizon_minutes=horizon_minutes,style=style,fee_bps=config['fee_bps'],
                            slippage_bps=config['slippage_bps'],initial_capital=config['initial_capital'],
                            max_folds=max_folds,model_family=model_family,
                            evaluation_start_minute=evaluation_start)
                        with conn.transaction():
                            conn.execute("update stock_engine.ml_backtest_runs set status=%s,metrics=%s where run_id=%s",
                                (result['status'],Jsonb(result['metrics']),run_id))
                            with conn.cursor() as cursor:
                                cursor.executemany("""insert into stock_engine.ml_backtest_predictions
                                    (run_id,origin_at,target_at,origin_price,actual_price,raw_return,predicted_return,actual_return,signal,model_id)
                                    values(%s,to_timestamp(%s*60),to_timestamp(%s*60),%s,%s,%s,%s,%s,%s,%s)""",
                                    [(run_id,p['origin_minute'],p['target_minute'],p['origin_price'],p['actual_price'],
                                      p['raw_return'],p['predicted_return'],p['actual_return'],p['signal'],p['model_id']) for p in result['predictions']])
                                cursor.executemany("""insert into stock_engine.ml_backtest_trades
                                    (run_id,origin_at,exit_at,side,entry_price,exit_price,gross_return,net_return,capital_before,capital_after)
                                    values(%s,to_timestamp(%s*60),to_timestamp(%s*60),%s,%s,%s,%s,%s,%s,%s)""",
                                    [(run_id,t['origin_minute'],t['exit_minute'],t['side'],t['entry_price'],t['exit_price'],
                                      t['gross_return'],t['net_return'],t['capital_before'],t['capital_after']) for t in result['trades']])
                            candidate=calibration_from_backtest(result['predictions'])
                            conn.execute("""insert into stock_engine.ml_backtest_calibration_candidates(run_id,candidate)
                                values(%s,%s)""",(run_id,Jsonb(candidate)))
                        # Only the live-compatible Ridge path can update calibration, and only after
                        # independent matured live-forward checks demonstrate improvement.
                        promotion=(
                            validate_and_promote(conn,stock,run_id,candidate,style=style,
                                window_minutes=window_minutes,horizon_minutes=horizon_minutes)
                            if model_family=='ridge'
                            and os.getenv('ALLOW_CALIBRATION_PROMOTION','false').lower()=='true'
                            else dict(promoted=False,reason=(
                                'promotion_disabled_for_backtest'
                                if os.getenv('ALLOW_CALIBRATION_PROMOTION','false').lower()!='true'
                                else 'alternative_family_not_wired_to_live_predictor'
                            ))
                        )
                        print(json.dumps(dict(stock_id=stock,run_id=str(run_id),model_family=model_family,config=config,
                            status=result['status'],metrics=result['metrics'],calibration_candidate=candidate,
                            calibration_promotion=promotion)))
            # Technical strategies are evaluated separately from ML regressors.
            # The same private shard tables are reused; no new grants or RLS changes.
            strategy_times = [int(value) * 60 for value in times]
            strategy_start = (int(evaluation_start * 60) if evaluation_start is not None else
                             strategy_times[int(len(strategy_times) * 0.70)])
            for horizon_minutes in (60, 180, 360, 720, 1080, 1440):
                competition = run_strategy_competition(
                    strategy_times, prices, horizon_steps=horizon_minutes,
                    bar_seconds=60, fee_bps=10, slippage_bps=5,
                    minimum_net_return=0.002, max_drawdown=0.15,
                    evaluation_start=strategy_start)
                for strategy_name, item in competition['strategies'].items():
                    if strategy_name.startswith('baseline_'):
                        continue
                    run_id = uuid.uuid4()
                    config = dict(style='short', window_minutes=lookback_days * 1440,
                        horizon_minutes=horizon_minutes, strategy_family=strategy_name,
                        model_family='technical_strategy', fee_bps=10,
                        slippage_bps=5, minimum_net_return=0.002,
                        execution=competition['config']['execution'],
                        risk_controls='0.5% equity risk budget; 15% position cap; 1.5x volatility stop; 2R target')
                    conn.execute(
                        "insert into stock_engine.ml_backtest_runs(run_id,stock_id,status,config,source_commit) values(%s,%s,'running',%s,%s)",
                        (run_id, stock, Jsonb(config), os.getenv('GITHUB_SHA')))
                    metrics = item['metrics']
                    with conn.transaction():
                        conn.execute(
                            "update stock_engine.ml_backtest_runs set status='completed',metrics=%s where run_id=%s",
                            (Jsonb(metrics), run_id))
                        with conn.cursor() as cursor:
                            cursor.executemany("""insert into stock_engine.ml_backtest_trades
                                (run_id,origin_at,exit_at,side,entry_price,exit_price,gross_return,net_return,capital_before,capital_after)
                                values(%s,to_timestamp(%s*60),to_timestamp(%s*60),'long',%s,%s,%s,%s,%s,%s)""",
                                [(run_id, int(trade['origin_time']//60), int(trade['exit_time']//60),
                                  trade['entry_price'], trade['exit_price'], trade['gross_return'],
                                  trade['net_return'], trade['capital_before'], trade['capital_after'])
                                 for trade in item['trades']])
                    print(json.dumps(dict(stock_id=stock, run_id=str(run_id),
                        strategy=strategy_name, horizon_minutes=horizon_minutes,
                        status=item['status'], metrics=metrics)))

def main():
    raw=os.getenv('STOCK_ML_PROJECTS_JSON','{}')
    try: projects=json.loads(raw)
    except ValueError: raise SystemExit('STOCK_ML_PROJECTS_JSON must be valid JSON')
    if not projects: raise SystemExit('Configure STOCK_ML_PROJECTS_JSON in repository secrets')
    requested={int(x) for x in os.getenv('STOCK_IDS','').split(',') if x.strip()}
    if requested and (min(requested)<1 or max(requested)>35): raise SystemExit('STOCK_IDS must be between 1 and 35')
    days=int(os.getenv('LOOKBACK_DAYS','180')); evaluation_days=int(os.getenv('EVALUATION_DAYS','0'))
    folds=int(os.getenv('MAX_FOLDS','30'))
    if not 30<=days<=180 or not 0<=evaluation_days<days or not 1<=folds<=500:
        raise SystemExit('lookback_days must be 30-180, evaluation_days 0 (all available) or a positive value less than lookback_days, max_folds 1-500')
    failed=False
    for n,dsn in projects.items():
        project=int(n)
        # Avoid opening unrelated shard connections during a targeted smoke test.
        if requested and not any(stock in (project*2-1, project*2) and stock<=35 for stock in requested):
            continue
        try: process(project,dsn,requested,days,evaluation_days,folds)
        except Exception:
            failed=True
            print(f"Shard {project} failed; full traceback follows:",file=sys.stderr)
            traceback.print_exc()
    return int(failed)
if __name__=='__main__': sys.exit(main())
