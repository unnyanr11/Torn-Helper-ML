"""Build backend-only rankings from recent completed walk-forward backtests."""
from __future__ import annotations
import json, os, sys, uuid, traceback
from collections import defaultdict
import psycopg
from psycopg.types.json import Jsonb

LOOKBACK_HOURS = 8
MIN_SAMPLES = 30
MIN_IMPROVEMENT = 0.02
MIN_TRADES = 5
MAX_DRAWDOWN = 0.15
MIN_PROFIT_FACTOR = 1.0


def rank_rows(rows):
    """Convert SQL aggregate rows into a compact, deterministic report."""
    evaluated=[]
    for row in rows:
        (stock_id, style, window_minutes, horizon_minutes, model_family,
         run_id, created_at, samples, mae, rmse, baseline_mae,
         directional_accuracy, mean_error, total_return, trades, win_rate,
         max_drawdown, profit_factor) = row
        samples=int(samples or 0)
        if not samples or mae is None or baseline_mae is None:
            continue
        mae=float(mae); baseline=float(baseline_mae)
        improvement=(baseline-mae)/baseline if baseline>0 else None
        total_return=float(total_return) if total_return is not None else None
        trades=int(trades or 0)
        max_drawdown=float(max_drawdown) if max_drawdown is not None else None
        profit_factor=float(profit_factor) if profit_factor is not None else None
        drawdown_abs=abs(max_drawdown) if max_drawdown is not None else None
        risk_adjusted_score=(total_return/max(drawdown_abs,0.01)
                             if total_return is not None and drawdown_abs is not None else None)
        eligible=(samples>=MIN_SAMPLES and improvement is not None
                  and improvement>=MIN_IMPROVEMENT and trades>=MIN_TRADES
                  and total_return is not None and total_return>0
                  and drawdown_abs is not None and drawdown_abs<=MAX_DRAWDOWN
                  and profit_factor is not None and profit_factor>MIN_PROFIT_FACTOR)
        evaluated.append(dict(
            stock_id=int(stock_id),style=style,window_minutes=int(window_minutes),
            horizon_minutes=int(horizon_minutes),model_family=model_family,
            run_id=str(run_id),created_at=created_at.isoformat() if created_at else None,
            samples=samples,mae=mae,rmse=float(rmse),baseline_mae=baseline,
            mae_improvement_pct=(improvement*100 if improvement is not None else None),
            directional_accuracy=(float(directional_accuracy) if directional_accuracy is not None else None),
            mean_error=(float(mean_error) if mean_error is not None else None),
            total_return=total_return,
            trades=trades,win_rate=(float(win_rate) if win_rate is not None else None),
            max_drawdown=max_drawdown,profit_factor=profit_factor,
            risk_adjusted_score=risk_adjusted_score,eligible=eligible))
    def quality_key(item):
        return (not item['eligible'],
                -(item['risk_adjusted_score'] if item['risk_adjusted_score'] is not None else -1e9),
                -(item['profit_factor'] if item['profit_factor'] is not None else -1e9),
                -(item['mae_improvement_pct'] or -1e9),
                -(item['directional_accuracy'] or 0), item['rmse'])
    evaluated.sort(key=lambda x: (
        x['stock_id'], x['style'], x['window_minutes'], x['horizon_minutes'],
        *quality_key(x)))
    eligible=[x for x in evaluated if x['eligible']]
    best_by_stock={}
    best_by_configuration={}
    for item in sorted(eligible,key=lambda x:(x['stock_id'],*quality_key(x))):
        best_by_stock.setdefault(item['stock_id'],item)
    for item in sorted(eligible,key=lambda x:(
            x['stock_id'],x['style'],x['window_minutes'],x['horizon_minutes'],*quality_key(x))):
        key=(item['stock_id'],item['style'],item['window_minutes'],item['horizon_minutes'])
        best_by_configuration.setdefault(key,item)
    return dict(
        lookback_hours=LOOKBACK_HOURS, minimum_samples=MIN_SAMPLES,
        minimum_mae_improvement_pct=MIN_IMPROVEMENT*100,
        minimum_trades=MIN_TRADES,maximum_drawdown_pct=MAX_DRAWDOWN*100,
        minimum_profit_factor=MIN_PROFIT_FACTOR,
        ranking_method='net_return_divided_by_max(absolute_drawdown, 1%) after eligibility filters',
        evaluated_runs=len(evaluated), eligible_runs=len(eligible),
        rejected_runs=len(evaluated)-len(eligible),
        best_model_by_stock=list(best_by_stock.values()),
        best_model_by_configuration=list(best_by_configuration.values()),
        rankings=evaluated)


def main():
    raw=os.getenv('STOCK_ML_PROJECTS_JSON','{}')
    try: projects=json.loads(raw)
    except ValueError: raise SystemExit('STOCK_ML_PROJECTS_JSON must be valid JSON')
    if not projects: raise SystemExit('Configure STOCK_ML_PROJECTS_JSON in repository secrets')
    failures=0
    for project,dsn in projects.items():
        try:
            with psycopg.connect(dsn,sslmode='require',connect_timeout=15,autocommit=True) as conn:
                settings=conn.execute("select project_number,enabled from stock_engine.settings where singleton").fetchone()
                if not settings or int(settings[0])!=int(project) or not settings[1]:
                    raise ValueError('project assignment mismatch or disabled project')
                rows=conn.execute("""
                    select r.stock_id,
                           r.config->>'style',
                           (r.config->>'window_minutes')::integer,
                           (r.config->>'horizon_minutes')::integer,
                           coalesce(r.config->>'model_family','ridge'),
                           r.run_id,r.created_at,count(p.run_id),
                           avg(abs(p.actual_return-p.predicted_return)),
                           sqrt(avg(power(p.actual_return-p.predicted_return,2))),
                           avg(abs(p.actual_return)),
                           avg(case when (p.predicted_return>=0)=(p.actual_return>=0) then 1.0 else 0.0 end),
                           avg(p.predicted_return-p.actual_return),
                           (r.metrics->>'total_return')::double precision,
                           (r.metrics->>'trades')::integer,
                           (r.metrics->>'win_rate')::double precision,
                           (r.metrics->>'max_drawdown')::double precision,
                           (r.metrics->>'profit_factor')::double precision
                    from stock_engine.ml_backtest_runs r
                    join stock_engine.ml_backtest_predictions p on p.run_id=r.run_id
                    where r.status='completed'
                      and r.created_at >= now()-(%s * interval '1 hour')
                    group by r.run_id,r.stock_id,r.config,r.created_at,r.metrics
                    order by r.stock_id,r.created_at desc
                """,(LOOKBACK_HOURS,)).fetchall()
                report=rank_rows(rows)
                report.update(project_number=int(project),generated_at=conn.execute("select now()").fetchone()[0].isoformat())
                report_id=uuid.uuid4()
                conn.execute("""
                    insert into stock_engine.ml_backtest_reports(report_id,report)
                    values(%s,%s)
                """,(report_id,Jsonb(report)))
                print(json.dumps(dict(event='backtest_report_saved',project=int(project),
                    report_id=str(report_id),evaluated_runs=report['evaluated_runs'],
                    eligible_runs=report['eligible_runs'],stocks_with_eligible_models=len(report['best_model_by_stock']))))
        except Exception:
            failures+=1
            print(f"Reporting failed for shard {project}",file=sys.stderr)
            traceback.print_exc()
    return 1 if failures else 0


if __name__=='__main__':
    sys.exit(main())
