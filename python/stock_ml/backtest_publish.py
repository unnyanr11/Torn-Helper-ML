"""Publish the best accuracy-validated walk-forward model for each live selector.

Backtest selection is accuracy-only: no fees, slippage, trading simulation or
profitability gate is used here. Calibration is fitted from the selected
family's chronological walk-forward predictions and accepted only on its
later holdout segment.
"""
import hashlib
import json
import os
import uuid
import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from backtesting import MODEL_FAMILIES, calibration_from_backtest, fit_live_model
from realtime_model import COMBINATIONS, validate_series

MIN_FOLDS = 30
MIN_MAE_IMPROVEMENT = 0.02
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))
PUBLISH_WINDOW_HOURS = int(os.getenv("BACKTEST_PUBLISH_WINDOW_HOURS", "8"))


def choose_winner(rows):
    """Choose the lowest-error eligible family on the shared walk-forward folds."""
    eligible=[]
    for row in rows:
        family=row["model_family"]
        folds=int(row["folds"] or 0)
        mae=float(row["mae"]) if row["mae"] is not None else None
        baseline=float(row["baseline_mae"]) if row["baseline_mae"] is not None else None
        if family not in MODEL_FAMILIES or mae is None or baseline is None or baseline<=0:
            continue
        improvement=(baseline-mae)/baseline
        directional=float(row["directional_accuracy"]) if row["directional_accuracy"] is not None else 0.0
        rmse=float(row["rmse"]) if row["rmse"] is not None else float("inf")
        if folds>=MIN_FOLDS and improvement>=MIN_MAE_IMPROVEMENT:
            eligible.append((mae,-directional,rmse,family,row,improvement))
    if not eligible:
        return None
    return min(eligible,key=lambda item:item[:4])


def publish_stock(conn, stock_id):
    query_rows=conn.execute("""
        select distinct on (config->>'style',(config->>'window_minutes')::integer,
                            (config->>'horizon_minutes')::integer,config->>'model_family')
               run_id,config->>'style' as style,
               (config->>'window_minutes')::integer as window_minutes,
               (config->>'horizon_minutes')::integer as horizon_minutes,
               config->>'model_family' as model_family,created_at,
               coalesce((metrics->>'folds')::integer,0) as folds,
               (metrics->'prediction_error'->>'mae')::double precision as mae,
               (metrics->'prediction_error'->>'rmse')::double precision as rmse,
               (metrics->'actual_return'->>'mae')::double precision as baseline_mae,
               (metrics->>'directional_accuracy')::double precision as directional_accuracy
        from stock_engine.ml_backtest_runs
        where stock_id=%s and status='completed'
          and created_at>=now()-(%s * interval '1 hour')
        order by config->>'style',(config->>'window_minutes')::integer,
                 (config->>'horizon_minutes')::integer,config->>'model_family',created_at desc
    """,(stock_id,PUBLISH_WINDOW_HOURS)).fetchall()
    by_config={}
    for r in query_rows:
        key=(r[1],int(r[2]),int(r[3]))
        by_config.setdefault(key,[]).append(dict(run_id=r[0],style=r[1],window_minutes=int(r[2]),
            horizon_minutes=int(r[3]),model_family=r[4],created_at=r[5],folds=r[6],
            mae=r[7],rmse=r[8],baseline_mae=r[9],directional_accuracy=r[10]))
    published=0
    for style,window,horizon in COMBINATIONS:
        rows=by_config.get((style,window,horizon),[])
        winner=choose_winner(rows)
        if winner is None:
            continue
        _,_,_,family,selected,improvement=winner
        predictions=conn.execute("""
            select floor(extract(epoch from origin_at)/60)::bigint,
                   raw_return,actual_return
            from stock_engine.ml_backtest_predictions
            where run_id=%s order by origin_at
        """,(selected["run_id"],)).fetchall()
        if len(predictions)<MIN_FOLDS:
            continue
        calibration_rows=[dict(origin_minute=int(p[0]),raw_return=float(p[1]),
                               predicted_return=float(p[1]),actual_return=float(p[2])) for p in predictions]
        candidate=calibration_from_backtest(calibration_rows,minimum_samples=MIN_FOLDS)
        price_rows=conn.execute("""
            select floor(extract(epoch from observed_at)/60)::bigint,
                   (array_agg(price order by observed_at desc))[1]
            from stock_engine.prices
            where stock_id=%s and observed_at>=now()-(%s * interval '1 day')
            group by 1 order by 1
        """,(stock_id,LOOKBACK_DAYS)).fetchall()
        if len(price_rows)<1000:
            continue
        times=np.asarray([int(r[0]) for r in price_rows],dtype=np.int64)
        prices=np.asarray([float(r[1]) for r in price_rows],dtype=float)
        times,prices=validate_series(times,prices)
        model=fit_live_model(times,prices,window_minutes=window,horizon_minutes=horizon,
                             model_family=family)
        residuals=np.asarray([float(p[2])-float(p[1]) for p in predictions],dtype=float)
        error_q90=float(np.quantile(np.abs(residuals),.90))
        model_id=hashlib.sha256(json.dumps({
            "stock_id":stock_id,"style":style,"window":window,"horizon":horizon,
            "family":family,"source_run":str(selected["run_id"]),
            "trained_through":model["trained_through"],
        },sort_keys=True).encode()).hexdigest()[:24]
        model.update(stock_id=stock_id,style=style,analysis_minutes=window,
            horizon_minutes=horizon,model_id=model_id,source="walk_forward_backtest_winner",
            source_backtest_run=str(selected["run_id"]),
            metrics={"approved":True,"mae":float(selected["mae"]),
                "baseline_mae":float(selected["baseline_mae"]),"rmse":float(selected["rmse"]),
                "error_q90":error_q90,"directional_accuracy":float(selected["directional_accuracy"]),
                "validation_rows":int(selected["folds"]),
                "mae_improvement_pct":round(improvement*100,4)})
        conn.execute("""
            insert into stock_engine.ml_v2_models
              (stock_id,style,analysis_minutes,horizon_minutes,payload,trained_at)
            values(%s,%s,%s,%s,%s,now())
            on conflict(stock_id,style,analysis_minutes,horizon_minutes)
            do update set payload=excluded.payload,trained_at=excluded.trained_at
        """,(stock_id,style,window,horizon,Jsonb(model)))
        # Calibration belongs to this exact selected model. Even a rejected
        # candidate replaces stale calibration with an inactive identity record.
        calibration_payload={"scale":float(candidate.get("scale",1.0)),
            "bias":float(candidate.get("bias",0.0)),
            "error_q90":(float(candidate["holdout_mae_corrected"]) if candidate.get("active") else None),
            "samples":int(candidate.get("samples",0)),"active":bool(candidate.get("active")),
            "model_id":model_id,"source":"selected_model_walk_forward_backtest",
            "source_backtest_run":str(selected["run_id"]),
            "holdout_mae_raw":candidate.get("holdout_mae_raw"),
            "holdout_mae_corrected":candidate.get("holdout_mae_corrected"),
            "holdout_start":candidate.get("holdout_start")}
        conn.execute("""
            insert into stock_engine.ml_v2_calibration
              (stock_id,style,analysis_minutes,horizon_minutes,payload,calibrated_at)
            values(%s,%s,%s,%s,%s,now())
            on conflict(stock_id,style,analysis_minutes,horizon_minutes)
            do update set payload=excluded.payload,calibrated_at=excluded.calibrated_at
        """,(stock_id,style,window,horizon,Jsonb(calibration_payload)))
        published+=1
        print(json.dumps({"event":"best_model_published","stock_id":stock_id,
            "style":style,"analysis_minutes":window,"horizon_minutes":horizon,
            "model_family":family,"model_id":model_id,"folds":selected["folds"],
            "mae_improvement_pct":round(improvement*100,3),
            "calibration_active":bool(candidate.get("active")),
            "calibration_samples":candidate.get("samples",0)}))
    return published


def main():
    projects=json.loads(os.getenv("STOCK_ML_PROJECTS_JSON","{}"))
    if not projects:
        raise SystemExit("Configure STOCK_ML_PROJECTS_JSON")
    requested={int(x) for x in os.getenv("STOCK_IDS","").split(",") if x.strip()}
    failed=False
    for project,dsn in projects.items():
        number=int(project)
        if requested and not any(s in (number*2-1,number*2) and s<=35 for s in requested):
            continue
        try:
            with psycopg.connect(dsn,sslmode="require",connect_timeout=15) as conn:
                assignment=conn.execute("select project_number,enabled from stock_engine.settings where singleton").fetchone()
                if not assignment or int(assignment[0])!=number or not assignment[1]:
                    raise ValueError("Stock shard assignment mismatch or disabled")
                stocks=[int(r[0]) for r in conn.execute("select stock_id from stock_engine.stocks order by stock_id").fetchall()]
                for stock in stocks:
                    if requested and stock not in requested:
                        continue
                    if stock not in (number*2-1,number*2) or stock>35:
                        raise ValueError("Unexpected stock assignment")
                    count=publish_stock(conn,stock)
                    print(json.dumps({"event":"stock_model_registry_updated","project":number,
                                      "stock_id":stock,"published_combinations":count}))
                conn.commit()
        except Exception as exc:
            failed=True
            print(json.dumps({"event":"model_publish_failed","project":number,
                              "error":type(exc).__name__}),flush=True)
    return int(failed)


if __name__=="__main__":
    raise SystemExit(main())
