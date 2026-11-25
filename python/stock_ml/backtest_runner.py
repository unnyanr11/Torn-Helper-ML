"""Run walk-forward backtests against each configured shard's observed prices."""
import json, os, sys, uuid
from datetime import datetime, timezone, timedelta
import psycopg
from psycopg.types.json import Jsonb
from backtesting import run_backtest, calibration_from_backtest

def process(project,dsn,stock_ids,lookback_days,max_folds):
    with psycopg.connect(dsn,sslmode='require',connect_timeout=15,autocommit=True) as conn:
        settings=conn.execute("select project_number,enabled from stock_engine.settings where singleton").fetchone()
        if not settings or int(settings[0])!=project or not settings[1]:
            raise ValueError('project assignment mismatch or disabled project')
        stocks=[int(r[0]) for r in conn.execute('select stock_id from stock_engine.stocks order by stock_id')]
        for stock in stocks:
            if stock not in (project*2-1,project*2) or stock>35: raise ValueError('unexpected stock assignment')
            if stock_ids and stock not in stock_ids: continue
            rows=conn.execute("""select floor(extract(epoch from observed_at)/60)::bigint,price
                from stock_engine.prices where stock_id=%s and observed_at>=now()-(%s * interval '1 day')
                order by observed_at""",(stock,lookback_days)).fetchall()
            if len(rows)<1000: print(f"stock {stock}: insufficient raw observations"); continue
            times=[int(r[0]) for r in rows]; prices=[float(r[1]) for r in rows]
            # Backtest a conservative 24-hour model with 7-day feature window;
            # other combinations can be added once history/cost profiles are known.
            run_id=uuid.uuid4()
            config=dict(style='short',window_minutes=10080,horizon_minutes=1440,
                fee_bps=10,slippage_bps=5,initial_capital=10000,max_folds=max_folds,
                lookback_days=lookback_days,execution='decision-close to exact-horizon close')
            conn.execute("insert into stock_engine.ml_backtest_runs(run_id,stock_id,status,config,source_commit) values(%s,%s,'running',%s,%s)",
                (run_id,stock,Jsonb(config),os.getenv('GITHUB_SHA')))
            result=run_backtest(times,prices,window_minutes=config['window_minutes'],
                horizon_minutes=config['horizon_minutes'],style=config['style'],
                fee_bps=config['fee_bps'],slippage_bps=config['slippage_bps'],
                initial_capital=config['initial_capital'],max_folds=max_folds)
            with conn.transaction():
                conn.execute("update stock_engine.ml_backtest_runs set status=%s,metrics=%s where run_id=%s",
                    (result['status'],Jsonb(result['metrics']),run_id))
                conn.executemany("""insert into stock_engine.ml_backtest_predictions
                    (run_id,origin_at,target_at,origin_price,actual_price,raw_return,predicted_return,actual_return,signal,model_id)
                    values(%s,to_timestamp(%s*60),to_timestamp(%s*60),%s,%s,%s,%s,%s,%s,%s)""",
                    [(run_id,p['origin_minute'],p['target_minute'],p['origin_price'],p['actual_price'],
                      p['raw_return'],p['predicted_return'],p['actual_return'],p['signal'],p['model_id']) for p in result['predictions']])
                conn.executemany("""insert into stock_engine.ml_backtest_trades
                    (run_id,origin_at,exit_at,side,entry_price,exit_price,gross_return,net_return,capital_before,capital_after)
                    values(%s,to_timestamp(%s*60),to_timestamp(%s*60),%s,%s,%s,%s,%s,%s,%s)""",
                    [(run_id,t['origin_minute'],t['exit_minute'],t['side'],t['entry_price'],t['exit_price'],
                      t['gross_return'],t['net_return'],t['capital_before'],t['capital_after']) for t in result['trades']])
                candidate=calibration_from_backtest(result['predictions'])
                conn.execute("""insert into stock_engine.ml_backtest_calibration_candidates(run_id,candidate)
                    values(%s,%s)""",(run_id,Jsonb(candidate)))
            print(json.dumps(dict(stock_id=stock,run_id=str(run_id),status=result['status'],metrics=result['metrics'],calibration_candidate=candidate)))
def main():
    raw=os.getenv('STOCK_ML_PROJECTS_JSON','{}')
    try: projects=json.loads(raw)
    except ValueError: raise SystemExit('STOCK_ML_PROJECTS_JSON must be valid JSON')
    if not projects: raise SystemExit('Configure STOCK_ML_PROJECTS_JSON in repository secrets')
    requested={int(x) for x in os.getenv('STOCK_IDS','').split(',') if x.strip()}
    if requested and (min(requested)<1 or max(requested)>35): raise SystemExit('STOCK_IDS must be between 1 and 35')
    days=int(os.getenv('LOOKBACK_DAYS','180')); folds=int(os.getenv('MAX_FOLDS','120'))
    if not 30<=days<=180 or not 10<=folds<=500: raise SystemExit('lookback_days must be 30-180 and max_folds 10-500')
    failed=False
    for n,dsn in projects.items():
        project=int(n)
        try: process(project,dsn,requested,days,folds)
        except Exception as exc:
            failed=True; print(f"shard {project} failed: {type(exc).__name__}",file=sys.stderr)
    return int(failed)
if __name__=='__main__': sys.exit(main())
